import * as fs from 'fs';
import * as jsonc from "jsonc-parser";
import * as path from 'path';
import * as vscode from 'vscode';

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
