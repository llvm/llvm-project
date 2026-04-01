import * as vscode from 'vscode';

import {AIIRContext} from '../aiirContext';
import {registerAIIRBytecodeExtensions} from './bytecodeProvider';

/**
 *  Register the necessary extensions for supporting AIIR.
 */
export function registerAIIRExtensions(context: vscode.ExtensionContext,
                                       aiirContext: AIIRContext) {
  registerAIIRBytecodeExtensions(context, aiirContext);
}
