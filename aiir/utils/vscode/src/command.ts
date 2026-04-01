import * as vscode from 'vscode';
import {AIIRContext} from './aiirContext';

/**
 * This class represents a base vscode command. It handles all of the necessary
 * command registration and disposal boilerplate.
 */
export abstract class Command extends vscode.Disposable {
  private disposable: vscode.Disposable;
  protected context: AIIRContext;

  constructor(command: string, context: AIIRContext) {
    super(() => this.dispose());
    this.disposable =
        vscode.commands.registerCommand(command, this.execute, this);
    this.context = context;
  }

  dispose() { this.disposable && this.disposable.dispose(); }

  /**
   * The function executed when this command is invoked.
   */
  abstract execute(...args: any[]): any;
}
