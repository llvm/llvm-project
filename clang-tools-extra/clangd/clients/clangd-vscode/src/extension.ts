import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';
import * as semanticHighlighting from './semantic-highlighting';

/**
 * Get an option from workspace configuration.
 * @param option name of the option (e.g. for clangd.path should be path)
 * @param defaultValue default value to return if option is not set
 */
function getConfig<T>(option: string, defaultValue?: any): T {
  const config = vscode.workspace.getConfiguration('clangd');
  return config.get<T>(option, defaultValue);
}

namespace SwitchSourceHeaderRequest {
export const type =
    new vscodelc.RequestType<vscodelc.TextDocumentIdentifier, string|undefined,
                             void, void>('textDocument/switchSourceHeader');
}

class FileStatus {
  private statuses = new Map<string, any>();
  private readonly statusBarItem =
      vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 10);

  onFileUpdated(fileStatus: any) {
    const filePath = vscode.Uri.parse(fileStatus.uri);
    this.statuses.set(filePath.fsPath, fileStatus);
    this.updateStatus();
  }

  updateStatus() {
    const path = vscode.window.activeTextEditor.document.fileName;
    const status = this.statuses.get(path);
    if (!status) {
      this.statusBarItem.hide();
      return;
    }
    this.statusBarItem.text = `clangd: ` + status.state;
    this.statusBarItem.show();
  }

  clear() {
    this.statuses.clear();
    this.statusBarItem.hide();
  }

  dispose() { this.statusBarItem.dispose(); }
}

class ClangdLanguageClient extends vscodelc.LanguageClient {
  // Override the default implementation for failed requests. The default
  // behavior is just to log failures in the output panel, however output panel
  // is designed for extension debugging purpose, normal users will not open it,
  // thus when the failure occurs, normal users doesn't know that.
  //
  // For user-interactive operations (e.g. applyFixIt, applyTweaks), we will
  // prompt up the failure to users.
  logFailedRequest(rpcReply: vscodelc.RPCMessageType, error: any) {
    if (error instanceof vscodelc.ResponseError &&
        rpcReply.method === "workspace/executeCommand")
      vscode.window.showErrorMessage(error.message);
    // Call default implementation.
    super.logFailedRequest(rpcReply, error);
  }
}

class EnableEditsNearCursorFeature implements vscodelc.StaticFeature {
  initialize() {}
  fillClientCapabilities(capabilities: vscodelc.ClientCapabilities): void {
    const extendedCompletionCapabilities: any =
        capabilities.textDocument.completion;
    extendedCompletionCapabilities.editsNearCursor = true;
  }
}

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const syncFileEvents = getConfig<boolean>('syncFileEvents', true);

  const clangd: vscodelc.Executable = {
    command : getConfig<string>('path'),
    args : getConfig<string[]>('arguments')
  };
  const traceFile = getConfig<string>('trace');
  if (!!traceFile) {
    const trace = {CLANGD_TRACE : traceFile};
    clangd.options = {env : {...process.env, ...trace}};
  }
  const serverOptions: vscodelc.ServerOptions = clangd;

  const clientOptions: vscodelc.LanguageClientOptions = {
        // Register the server for c-family and cuda files.
        documentSelector: [
            { scheme: 'file', language: 'c' },
            { scheme: 'file', language: 'cpp' },
            // CUDA is not supported by vscode, but our extension does supports it.
            { scheme: 'file', language: 'cuda' },
            { scheme: 'file', language: 'objective-c'},
            { scheme: 'file', language: 'objective-cpp'}
        ],
        synchronize: !syncFileEvents ? undefined : {
        // FIXME: send sync file events when clangd provides implementations.
        },
        initializationOptions: { clangdFileStatus: true },
        // Do not switch to output window when clangd returns output.
        revealOutputChannelOn: vscodelc.RevealOutputChannelOn.Never,

        // We hack up the completion items a bit to prevent VSCode from re-ranking them
        // and throwing away all our delicious signals like type information.
        //
        // VSCode sorts by (fuzzymatch(prefix, item.filterText), item.sortText)
        // By adding the prefix to the beginning of the filterText, we get a perfect
        // fuzzymatch score for every item.
        // The sortText (which reflects clangd ranking) breaks the tie.
        // This also prevents VSCode from filtering out any results due to the
        // differences in how fuzzy filtering is applies, e.g. enable dot-to-arrow
        // fixes in completion.
        //
        // We also have to mark the list as incomplete to force retrieving new rankings.
        // See https://github.com/microsoft/language-server-protocol/issues/898
        middleware: {
          provideCompletionItem: async (document, position, context, token, next) => {
            let list = await next(document, position, context, token);
            let items = (Array.isArray(list) ? list : list.items).map(item => {
              // Gets the prefix used by VSCode when doing fuzzymatch.
              let prefix = document.getText(new vscode.Range(item.range.start, position))
              if (prefix)
                item.filterText = prefix + "_" + item.filterText;
              return item;
            })
            return new vscode.CompletionList(items, /*isIncomplete=*/true);
          }
        },
    };

  const clangdClient = new ClangdLanguageClient('Clang Language Server',
                                                serverOptions, clientOptions);
  if (getConfig<boolean>('semanticHighlighting')) {
    const semanticHighlightingFeature =
        new semanticHighlighting.SemanticHighlightingFeature(clangdClient,
                                                             context);
    context.subscriptions.push(
        vscode.Disposable.from(semanticHighlightingFeature));
    clangdClient.registerFeature(semanticHighlightingFeature);
  }
  clangdClient.registerFeature(new EnableEditsNearCursorFeature);
  console.log('Clang Language Server is now active!');
  context.subscriptions.push(clangdClient.start());
  context.subscriptions.push(vscode.commands.registerCommand(
      'clangd-vscode.switchheadersource', async () => {
        const uri =
            vscode.Uri.file(vscode.window.activeTextEditor.document.fileName);
        if (!uri) {
          return;
        }
        const docIdentifier =
            vscodelc.TextDocumentIdentifier.create(uri.toString());
        const sourceUri = await clangdClient.sendRequest(
            SwitchSourceHeaderRequest.type, docIdentifier);
        if (!sourceUri) {
          return;
        }
        const doc = await vscode.workspace.openTextDocument(
            vscode.Uri.parse(sourceUri));
        vscode.window.showTextDocument(doc);
      }));
  const status = new FileStatus();
  context.subscriptions.push(vscode.Disposable.from(status));
  context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(
      () => { status.updateStatus(); }));
  context.subscriptions.push(clangdClient.onDidChangeState(({newState}) => {
    if (newState == vscodelc.State.Running) {
      // clangd starts or restarts after crash.
      clangdClient.onNotification(
          'textDocument/clangd.fileStatus',
          (fileStatus) => { status.onFileUpdated(fileStatus); });
    } else if (newState == vscodelc.State.Stopped) {
      // Clear all cached statuses when clangd crashes.
      status.clear();
    }
  }));
  // An empty place holder for the activate command, otherwise we'll get an
  // "command is not registered" error.
  context.subscriptions.push(vscode.commands.registerCommand(
      'clangd-vscode.activate', async () => {}));
}
