import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

/**
 * Method to get workspace configuration option
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

    dispose() {
        this.statusBarItem.dispose();
    }
}

/**
 *  this method is called when your extension is activate
 *  your extension is activated the very first time the command is executed
 */
export function activate(context: vscode.ExtensionContext) {
    const syncFileEvents = getConfig<boolean>('syncFileEvents', true);

    const clangd: vscodelc.Executable = {
        command: getConfig<string>('path'),
        args: getConfig<string[]>('arguments')
    };
    const traceFile = getConfig<string>('trace');
    if (!!traceFile) {
        const trace = { CLANGD_TRACE: traceFile };
        clangd.options = { env: { ...process.env, ...trace } };
    }
    const serverOptions: vscodelc.ServerOptions = clangd;

    // Note that CUDA ('.cu') files are special. When opening files of all other
    // extensions, VSCode would load clangd automatically. This is achieved by
    // having a corresponding 'onLanguage:...' activation event in package.json.
    // However, VSCode does not have CUDA as a supported language yet, so we
    // cannot add a corresponding activationEvent for CUDA files and clangd will
    // *not* load itself automatically on '.cu' files. When any of the files
    // with other extensions are open, clangd will load itself and will also
    // work on '.cu' files.
    const filePattern: string = '**/*.{' +
        ['cpp', 'c', 'cc', 'cu', 'cxx', 'c++', 'm', 'mm',
            'h', 'hh', 'hpp', 'hxx', 'inc'].join()
        + '}';
    const clientOptions: vscodelc.LanguageClientOptions = {
        // Register the server for C/C++ files
        documentSelector: [{ scheme: 'file', pattern: filePattern }],
        synchronize: !syncFileEvents ? undefined : {
            fileEvents: vscode.workspace.createFileSystemWatcher(filePattern)
        },
        initializationOptions: { clangdFileStatus: true },
        // Do not switch to output window when clangd returns output
        revealOutputChannelOn: vscodelc.RevealOutputChannelOn.Never
    };

    const clangdClient = new vscodelc.LanguageClient('Clang Language Server',serverOptions, clientOptions);
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
    context.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(() => {
        status.updateStatus();
    }));
    clangdClient.onDidChangeState(
        ({ newState }) => {
            if (newState == vscodelc.State.Running) {
                // clangd starts or restarts after crash.
                clangdClient.onNotification(
                    'textDocument/clangd.fileStatus',
                    (fileStatus) => { status.onFileUpdated(fileStatus); });
            } else if (newState == vscodelc.State.Stopped) {
                // Clear all cached statuses when clangd crashes.
                status.clear();
            }
        })
    // An empty place holder for the activate command, otherwise we'll get an
    // "command is not registered" error.
    context.subscriptions.push(vscode.commands.registerCommand(
            'clangd-vscode.activate', async () => {}));
}
