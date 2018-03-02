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
    const syncFileEvents = getConfig<boolean>('syncFileEvents', true);

    const clangd: vscodelc.Executable = {
        command: getConfig<string>('path'),
        args: getConfig<string[]>('arguments')
    };
    const traceFile = getConfig<string>('trace');
    if (!!traceFile) {
      const trace = {CLANGD_TRACE : traceFile};
      clangd.options = {env : {...process.env, ...trace}};
    }
    const serverOptions: vscodelc.ServerOptions = clangd;

    const filePattern: string = '**/*.{' +
      ['cpp', 'c', 'cc', 'cxx', 'c++', 'm', 'mm', 'h', 'hh', 'hpp', 'hxx', 'inc'].join() + '}';
    const clientOptions: vscodelc.LanguageClientOptions = {
        // Register the server for C/C++ files
        documentSelector: [{scheme: 'file', pattern: filePattern}],
        synchronize: !syncFileEvents ? undefined : {
            fileEvents: vscode.workspace.createFileSystemWatcher(filePattern)
        }
    };

    const clangdClient = new vscodelc.LanguageClient('Clang Language Server', serverOptions, clientOptions);
    console.log('Clang Language Server is now active!');

    const disposable = clangdClient.start();
}
