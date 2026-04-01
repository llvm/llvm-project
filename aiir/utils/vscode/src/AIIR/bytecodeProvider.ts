import * as base64 from 'base64-js'
import * as vscode from 'vscode'

import {AIIRContext} from '../aiirContext';

/**
 * The parameters to the aiir/convert(To|From)Bytecode commands. These
 * parameters are:
 * - `uri`: The URI of the file to convert.
 */
type ConvertBytecodeParams = Partial<{uri : string}>;

/**
 * The output of the aiir/convert(To|From)Bytecode commands:
 * - `output`: The output buffer of the command, e.g. a .aiir or bytecode
 *             buffer.
 */
type ConvertBytecodeResult = Partial<{output : string}>;

/**
 * A custom filesystem that is used to convert AIIR bytecode files to text for
 * use in the editor, but still use bytecode on disk.
 */
class BytecodeFS implements vscode.FileSystemProvider {
  aiirContext: AIIRContext;

  constructor(aiirContext: AIIRContext) { this.aiirContext = aiirContext; }

  /*
   * Forward to the default filesystem for the various methods that don't need
   * to understand the bytecode <-> text translation.
   */
  readDirectory(uri: vscode.Uri): Thenable<[ string, vscode.FileType ][]> {
    return vscode.workspace.fs.readDirectory(uri);
  }
  delete(uri: vscode.Uri): void {
    vscode.workspace.fs.delete(uri.with({scheme : "file"}));
  }
  stat(uri: vscode.Uri): Thenable<vscode.FileStat> {
    return vscode.workspace.fs.stat(uri.with({scheme : "file"}));
  }
  rename(oldUri: vscode.Uri, newUri: vscode.Uri,
         options: {overwrite: boolean}): void {
    vscode.workspace.fs.rename(oldUri.with({scheme : "file"}),
                               newUri.with({scheme : "file"}), options);
  }
  createDirectory(uri: vscode.Uri): void {
    vscode.workspace.fs.createDirectory(uri.with({scheme : "file"}));
  }
  watch(_uri: vscode.Uri, _options: {
    readonly recursive: boolean; readonly excludes : readonly string[]
  }): vscode.Disposable {
    return new vscode.Disposable(() => {});
  }

  private _emitter = new vscode.EventEmitter<vscode.FileChangeEvent[]>();
  readonly onDidChangeFile: vscode.Event<vscode.FileChangeEvent[]> =
      this._emitter.event;

  /*
   * Read in a bytecode file, converting it to text before returning it to the
   * caller.
   */
  async readFile(uri: vscode.Uri): Promise<Uint8Array> {
    // Try to start a language client for this file so that we can parse
    // it.
    const client =
        await this.aiirContext.getOrActivateLanguageClient(uri, 'aiir');
    if (!client) {
      throw new Error(
          'Failed to activate aiir language server to read bytecode');
    }

    // Ask the client to do the conversion.
    let result: ConvertBytecodeResult;
    try {
      let params: ConvertBytecodeParams = {uri : uri.toString()};
      result = await client.sendRequest('aiir/convertFromBytecode', params);
    } catch (e) {
      vscode.window.showErrorMessage(e.message);
      throw new Error(`Failed to read bytecode file: ${e}`);
    }
    let resultBuffer = new TextEncoder().encode(result.output);

    // NOTE: VSCode does not allow for extensions to manage files above 50mb.
    // Detect that here and if our result is too large for us to manage, alert
    // the user and open it as a new temporary .aiir file.
    if (resultBuffer.length > (50 * 1024 * 1024)) {
      const openAsTempInstead: vscode.MessageItem = {
        title : 'Open as temporary .aiir instead',
      };
      const message: string = `Failed to open bytecode file "${
          uri.toString()}". Cannot edit converted bytecode files larger than 50MB.`;
      const errorResult: vscode.MessageItem|undefined =
          await vscode.window.showErrorMessage(message, openAsTempInstead);
      if (errorResult === openAsTempInstead) {
        let tempFile = await vscode.workspace.openTextDocument({
          language : 'aiir',
          content : result.output,
        });
        await vscode.window.showTextDocument(tempFile);
      }
      throw new Error(message);
    }

    return resultBuffer;
  }

  /*
   * Save the provided content, which contains AIIR text, as bytecode.
   */
  async writeFile(uri: vscode.Uri, content: Uint8Array,
                  _options: {create: boolean, overwrite: boolean}) {
    // Get the language client managing this file.
    let client = this.aiirContext.getLanguageClient(uri, 'aiir');
    if (!client) {
      throw new Error(
          'Failed to activate aiir language server to write bytecode');
    }

    // Ask the client to do the conversion.
    let convertParams: ConvertBytecodeParams = {
      uri : uri.toString(),
    };
    const result: ConvertBytecodeResult =
        await client.sendRequest('aiir/convertToBytecode', convertParams);
    await vscode.workspace.fs.writeFile(uri.with({scheme : "file"}),
                                        base64.toByteArray(result.output));
  }
}

/**
 * A custom bytecode document for use by the custom editor provider below.
 */
class BytecodeDocument implements vscode.CustomDocument {
  readonly uri: vscode.Uri;

  constructor(uri: vscode.Uri) { this.uri = uri; }
  dispose(): void {}
}

/**
 * A custom editor provider for AIIR bytecode that allows for non-binary
 * interpretation.
 */
class BytecodeEditorProvider implements
    vscode.CustomReadonlyEditorProvider<BytecodeDocument> {
  public async openCustomDocument(uri: vscode.Uri, _openContext: any,
                                  _token: vscode.CancellationToken):
      Promise<BytecodeDocument> {
    return new BytecodeDocument(uri);
  }

  public async resolveCustomEditor(document: BytecodeDocument,
                                   _webviewPanel: vscode.WebviewPanel,
                                   _token: vscode.CancellationToken):
      Promise<void> {
    // Ask the user for the desired view type.
    const editType = await vscode.window.showQuickPick(
        [ {label : '.aiir', description : "Edit as a .aiir text file"} ],
        {title : 'Select an editor for the bytecode.'},
    );

    // If we don't have a valid view type, just bail.
    if (!editType) {
      await vscode.commands.executeCommand(
          'workbench.action.closeActiveEditor');
      return;
    }

    // TODO: We should also provide a non-`.aiir` way of viewing the
    // bytecode, which should also ideally have some support for invalid
    // bytecode files.

    // Close the active editor given that we aren't using it.
    await vscode.commands.executeCommand('workbench.action.closeActiveEditor');

    // Display the file using a .aiir format.
    await vscode.window.showTextDocument(
        document.uri.with({scheme : "aiir.bytecode-aiir"}),
        {preview : true, preserveFocus : false});
  }
}

/**
 *  Register the necessary providers for supporting AIIR bytecode.
 */
export function registerAIIRBytecodeExtensions(context: vscode.ExtensionContext,
                                               aiirContext: AIIRContext) {
  vscode.workspace.registerFileSystemProvider("aiir.bytecode-aiir",
                                              new BytecodeFS(aiirContext));
  vscode.window.registerCustomEditorProvider('aiir.bytecode',
                                             new BytecodeEditorProvider());
}
