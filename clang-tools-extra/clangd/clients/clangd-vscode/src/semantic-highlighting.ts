import * as fs from 'fs';
import * as jsonc from "jsonc-parser";
import * as path from 'path';
import * as vscode from 'vscode';
import * as vscodelc from 'vscode-languageclient';
import * as vscodelct from 'vscode-languageserver-types';

// Parameters for the semantic highlighting (server-side) push notification.
// Mirrors the structure in the semantic highlighting proposal for LSP.
interface SemanticHighlightingParams {
  // The text document that has to be decorated with the semantic highlighting
  // information.
  textDocument: vscodelct.VersionedTextDocumentIdentifier;
  // An array of semantic highlighting information.
  lines: SemanticHighlightingInformation[];
}
// Contains the highlighting information for a specified line. Mirrors the
// structure in the semantic highlighting proposal for LSP.
interface SemanticHighlightingInformation {
  // The zero-based line position in the text document.
  line: number;
  // A base64 encoded string representing every single highlighted characters
  // with its start position, length and the "lookup table" index of of the
  // semantic highlighting Text Mate scopes.
  tokens?: string;
}

// A SemanticHighlightingToken decoded from the base64 data sent by clangd.
interface SemanticHighlightingToken {
  // Start column for this token.
  character: number;
  // Length of the token.
  length: number;
  // The TextMate scope index to the clangd scope lookup table.
  scopeIndex: number;
}

// Language server push notification providing the semantic highlighting
// information for a text document.
export const NotificationType =
    new vscodelc.NotificationType<SemanticHighlightingParams, void>(
        'textDocument/semanticHighlighting');

// The feature that should be registered in the vscode lsp for enabling
// experimental semantic highlighting.
export class SemanticHighlightingFeature implements vscodelc.StaticFeature {
  // The TextMate scope lookup table. A token with scope index i has the scopes
  // on index i in the lookup table.
  scopeLookupTable: string[][];
  fillClientCapabilities(capabilities: vscodelc.ClientCapabilities) {
    // Extend the ClientCapabilities type and add semantic highlighting
    // capability to the object.
    const textDocumentCapabilities: vscodelc.TextDocumentClientCapabilities&
        {semanticHighlightingCapabilities?: {semanticHighlighting : boolean}} =
        capabilities.textDocument;
    textDocumentCapabilities.semanticHighlightingCapabilities = {
      semanticHighlighting : true,
    };
  }

  initialize(capabilities: vscodelc.ServerCapabilities,
             documentSelector: vscodelc.DocumentSelector|undefined) {
    // The semantic highlighting capability information is in the capabilities
    // object but to access the data we must first extend the ServerCapabilities
    // type.
    const serverCapabilities: vscodelc.ServerCapabilities&
        {semanticHighlighting?: {scopes : string[][]}} = capabilities;
    if (!serverCapabilities.semanticHighlighting)
      return;
    this.scopeLookupTable = serverCapabilities.semanticHighlighting.scopes;
  }

  handleNotification(params: SemanticHighlightingParams) {}
}

// Converts a string of base64 encoded tokens into the corresponding array of
// HighlightingTokens.
export function decodeTokens(tokens: string): SemanticHighlightingToken[] {
  const scopeMask = 0xFFFF;
  const lenShift = 0x10;
  const uint32Size = 4;
  const buf = Buffer.from(tokens, 'base64');
  const retTokens = [];
  for (let i = 0, end = buf.length / uint32Size; i < end; i += 2) {
    const start = buf.readUInt32BE(i * uint32Size);
    const lenKind = buf.readUInt32BE((i + 1) * uint32Size);
    const scopeIndex = lenKind & scopeMask;
    const len = lenKind >>> lenShift;
    retTokens.push({character : start, scopeIndex : scopeIndex, length : len});
  }

  return retTokens;
}

// A rule for how to color TextMate scopes.
interface TokenColorRule {
  // A TextMate scope that specifies the context of the token, e.g.
  // "entity.name.function.cpp".
  scope: string;
  // foreground is the color tokens of this scope should have.
  foreground: string;
}

// Get all token color rules provided by the theme.
function loadTheme(themeName: string): Promise<TokenColorRule[]> {
  const extension =
      vscode.extensions.all.find((extension: vscode.Extension<any>) => {
        const contribs = extension.packageJSON.contributes;
        if (!contribs || !contribs.themes)
          return false;
        return contribs.themes.some((theme: any) => theme.id === themeName ||
                                                    theme.label === themeName);
      });

  if (!extension) {
    return Promise.reject('Could not find a theme with name: ' + themeName);
  }

  const themeInfo = extension.packageJSON.contributes.themes.find(
      (theme: any) => theme.id === themeName || theme.label === themeName);
  return parseThemeFile(path.join(extension.extensionPath, themeInfo.path));
}

/**
 * Parse the TextMate theme at fullPath. If there are multiple TextMate scopes
 * of the same name in the include chain only the earliest entry of the scope is
 * saved.
 * @param fullPath The absolute path to the theme.
 * @param seenScopes A set containing the name of the scopes that have already
 *     been set.
 */
export async function parseThemeFile(
    fullPath: string, seenScopes?: Set<string>): Promise<TokenColorRule[]> {
  if (!seenScopes)
    seenScopes = new Set();
  // FIXME: Add support for themes written as .tmTheme.
  if (path.extname(fullPath) === '.tmTheme')
    return [];
  try {
    const contents = await readFileText(fullPath);
    const parsed = jsonc.parse(contents);
    const rules: TokenColorRule[] = [];
    // To make sure it does not crash if tokenColors is undefined.
    if (!parsed.tokenColors)
      parsed.tokenColors = [];
    parsed.tokenColors.forEach((rule: any) => {
      if (!rule.scope || !rule.settings || !rule.settings.foreground)
        return;
      const textColor = rule.settings.foreground;
      // Scopes that were found further up the TextMate chain should not be
      // overwritten.
      const addColor = (scope: string) => {
        if (seenScopes.has(scope))
          return;
        rules.push({scope, foreground : textColor});
        seenScopes.add(scope);
      };
      if (rule.scope instanceof Array) {
        return rule.scope.forEach((s: string) => addColor(s));
      }
      addColor(rule.scope);
    });

    if (parsed.include)
      // Get all includes and merge into a flat list of parsed json.
      return [
        ...(await parseThemeFile(
            path.join(path.dirname(fullPath), parsed.include), seenScopes)),
        ...rules
      ];
    return rules;
  } catch (err) {
    // If there is an error opening a file, the TextMate files that were
    // correctly found and parsed further up the chain should be returned.
    // Otherwise there will be no highlightings at all.
    console.warn('Could not open file: ' + fullPath + ', error: ', err);
  }

  return [];
}

function readFileText(path: string): Promise<string> {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, data) => {
      if (err) {
        return reject(err);
      }
      return resolve(data);
    });
  });
}
