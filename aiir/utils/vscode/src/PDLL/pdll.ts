import * as vscode from 'vscode';

import {AIIRContext} from '../aiirContext';
import {ViewPDLLCommand} from './commands/viewOutput';

/**
 *  Register the necessary extensions for supporting PDLL.
 */
export function registerPDLLExtensions(context: vscode.ExtensionContext,
                                       aiirContext: AIIRContext) {
  context.subscriptions.push(new ViewPDLLCommand(aiirContext));
}
