import * as vscode from 'vscode';

import {registerAIIRExtensions} from './AIIR/aiir';
import {AIIRContext} from './aiirContext';
import {registerPDLLExtensions} from './PDLL/pdll';

/**
 *  This method is called when the extension is activated. The extension is
 *  activated the very first time a command is executed.
 */
export function activate(context: vscode.ExtensionContext) {
  const outputChannel = vscode.window.createOutputChannel('AIIR');
  context.subscriptions.push(outputChannel);

  const aiirContext = new AIIRContext();
  context.subscriptions.push(aiirContext);

  // Initialize the commands of the extension.
  context.subscriptions.push(
      vscode.commands.registerCommand('aiir.restart', async () => {
        // Dispose and reactivate the context.
        aiirContext.dispose();
        await aiirContext.activate(outputChannel);
      }));
  registerAIIRExtensions(context, aiirContext);
  registerPDLLExtensions(context, aiirContext);

  aiirContext.activate(outputChannel);
}
