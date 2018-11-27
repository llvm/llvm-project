import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';
import { realpathSync } from 'fs';

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

    const filePattern: string = '**/*.{' +
        ['cpp', 'c', 'cc', 'cxx', 'c++', 'm', 'mm', 'h', 'hh', 'hpp', 'hxx', 'inc'].join() + '}';
    const clientOptions: vscodelc.LanguageClientOptions = {
        // Register the server for C/C++ files
        documentSelector: [{ scheme: 'file', pattern: filePattern }],
        synchronize: !syncFileEvents ? undefined : {
            fileEvents: vscode.workspace.createFileSystemWatcher(filePattern)
        },
        // Resolve symlinks for all files provided by clangd.
        // This is a workaround for a bazel + clangd issue - bazel produces a symlink tree to build in,
        // and when navigating to the included file, clangd passes its path inside the symlink tree
        // rather than its filesystem path.
        // FIXME: remove this once clangd knows enough about bazel to resolve the
        // symlinks where needed (or if this causes problems for other workflows).
        uriConverters: {
            code2Protocol: (value: vscode.Uri) => value.toString(),
            protocol2Code: (value: string) =>
                vscode.Uri.file(realpathSync(vscode.Uri.parse(value).fsPath))
        }
    };

  const clangdClient = new vscodelc.LanguageClient('Clang Language Server', serverOptions, clientOptions);
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
}
