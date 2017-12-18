import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';

/**
 * Method to get workspace configuration option
 * @param option name of the option (e.g. for clangd.path should be path)
 * @param defaultValue default value to return if option is not set
 */
function getConfig<T>(option: string, defaultValue?: any) : T {
    const config = vscode.workspace.getConfiguration('clangd');
    return config.get<T>(option, defaultValue);
}

/**
 *  this method is called when your extension is activate
 *  your extension is activated the very first time the command is executed
 */
export function activate(context: vscode.ExtensionContext) {
    const clangdPath = getConfig<string>('path');
    const clangdArgs = getConfig<string[]>('arguments');
    const syncFileEvents = getConfig<boolean>('syncFileEvents', true);

    const serverOptions: vscodelc.ServerOptions = { command: clangdPath, args: clangdArgs };

    const filePattern: string = '**/*.{' +
      ['cpp', 'c', 'cc', 'cxx', 'c++', 'm', 'mm', 'h', 'hh', 'hpp', 'hxx', 'inc'].join() + '}';
    const clientOptions: vscodelc.LanguageClientOptions = {
        // Register the server for C/C++ files
        documentSelector: [{scheme: 'file', pattern: filePattern}],
        uriConverters: {
            // FIXME: by default the URI sent over the protocol will be percent encoded (see rfc3986#section-2.1)
            //        the "workaround" below disables temporarily the encoding until decoding
            //        is implemented properly in clangd
            code2Protocol: (uri: vscode.Uri) : string => uri.toString(true),
            protocol2Code: (uri: string) : vscode.Uri => vscode.Uri.parse(uri)
        },
        synchronize: !syncFileEvents ? undefined : {
            fileEvents: vscode.workspace.createFileSystemWatcher(filePattern)
        }
    };

    const clangdClient = new vscodelc.LanguageClient('Clang Language Server', serverOptions, clientOptions);
    console.log('Clang Language Server is now active!');

    const disposable = clangdClient.start();
}
