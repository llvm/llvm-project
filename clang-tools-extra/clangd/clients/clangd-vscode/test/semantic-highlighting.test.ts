import * as assert from 'assert';
import * as path from 'path';
import * as vscode from 'vscode';

import * as semanticHighlighting from '../src/semantic-highlighting';

suite('SemanticHighlighting Tests', () => {
  test('Parses arrays of textmate themes.', async () => {
    const themePath =
        path.join(__dirname, '../../test/assets/includeTheme.jsonc');
    const scopeColorRules =
        await semanticHighlighting.parseThemeFile(themePath);
    const getScopeRule = (scope: string) =>
        scopeColorRules.find((v) => v.scope === scope);
    assert.equal(scopeColorRules.length, 3);
    assert.deepEqual(getScopeRule('a'), {scope : 'a', foreground : '#fff'});
    assert.deepEqual(getScopeRule('b'), {scope : 'b', foreground : '#000'});
    assert.deepEqual(getScopeRule('c'), {scope : 'c', foreground : '#bcd'});
  });
  test('Decodes tokens correctly', () => {
    const testCases: string[] = [
      'AAAAAAABAAA=', 'AAAAAAADAAkAAAAEAAEAAA==',
      'AAAAAAADAAkAAAAEAAEAAAAAAAoAAQAA'
    ];
    const expected = [
      [ {character : 0, scopeIndex : 0, length : 1} ],
      [
        {character : 0, scopeIndex : 9, length : 3},
        {character : 4, scopeIndex : 0, length : 1}
      ],
      [
        {character : 0, scopeIndex : 9, length : 3},
        {character : 4, scopeIndex : 0, length : 1},
        {character : 10, scopeIndex : 0, length : 1}
      ]
    ];
    testCases.forEach(
        (testCase, i) => assert.deepEqual(
            semanticHighlighting.decodeTokens(testCase), expected[i]));
  });
  test('ScopeRules overrides for more specific themes', () => {
    const rules = [
      {scope : 'variable.other.css', foreground : '1'},
      {scope : 'variable.other', foreground : '2'},
      {scope : 'storage', foreground : '3'},
      {scope : 'storage.static', foreground : '4'},
      {scope : 'storage', foreground : '5'},
      {scope : 'variable.other.parameter', foreground : '6'},
    ];
    const tm = new semanticHighlighting.ThemeRuleMatcher(rules);
    assert.deepEqual(tm.getBestThemeRule('variable.other.cpp').scope,
                     'variable.other');
    assert.deepEqual(tm.getBestThemeRule('storage.static').scope,
                     'storage.static');
    assert.deepEqual(
        tm.getBestThemeRule('storage'),
        rules[2]); // Match the first element if there are duplicates.
    assert.deepEqual(tm.getBestThemeRule('variable.other.parameter').scope,
                     'variable.other.parameter');
    assert.deepEqual(tm.getBestThemeRule('variable.other.parameter.cpp').scope,
                     'variable.other.parameter');
  });
  test('Colorizer groups decorations correctly', async () => {
    const scopeTable = [
      [ 'variable' ], [ 'entity.type.function' ],
      [ 'entity.type.function.method' ]
    ];
    // Create the scope source ranges the highlightings should be highlighted
    // at. Assumes the scopes used are the ones in the "scopeTable" variable.
    const createHighlightingScopeRanges =
        (highlightingLines:
             semanticHighlighting.SemanticHighlightingLine[]) => {
          // Initialize the scope ranges list to the correct size. Otherwise
          // scopes that don't have any highlightings are missed.
          let scopeRanges: vscode.Range[][] = scopeTable.map(() => []);
          highlightingLines.forEach((line) => {
            line.tokens.forEach((token) => {
              scopeRanges[token.scopeIndex].push(new vscode.Range(
                  new vscode.Position(line.line, token.character),
                  new vscode.Position(line.line,
                                      token.character + token.length)));
            });
          });
          return scopeRanges;
        };

    const fileUri1 = vscode.Uri.parse('file:///file1');
    const fileUri2 = vscode.Uri.parse('file:///file2');
    const fileUri1Str = fileUri1.toString();
    const fileUri2Str = fileUri2.toString();

    class MockHighlighter extends semanticHighlighting.Highlighter {
      applicationUriHistory: string[] = [];
      // Override to make the highlighting calls accessible to the test. Also
      // makes the test not depend on visible text editors.
      applyHighlights(fileUri: vscode.Uri) {
        this.applicationUriHistory.push(fileUri.toString());
      }
      // Override to make it accessible from the test.
      getDecorationRanges(fileUri: vscode.Uri) {
        return super.getDecorationRanges(fileUri);
      }
      // Override to make tests not depend on visible text editors.
      getVisibleTextEditorUris() { return [ fileUri1, fileUri2 ]; }
    }
    const highlighter = new MockHighlighter(scopeTable);
    const tm = new semanticHighlighting.ThemeRuleMatcher([
      {scope : 'variable', foreground : '1'},
      {scope : 'entity.type', foreground : '2'},
    ]);
    // Recolorizes when initialized.
    highlighter.highlight(fileUri1, []);
    assert.deepEqual(highlighter.applicationUriHistory, [ fileUri1Str ]);
    highlighter.initialize(tm);
    assert.deepEqual(highlighter.applicationUriHistory,
                     [ fileUri1Str, fileUri1Str, fileUri2Str ]);
    // Groups decorations into the scopes used.
    let highlightingsInLine: semanticHighlighting.SemanticHighlightingLine[] = [
      {
        line : 1,
        tokens : [
          {character : 1, length : 2, scopeIndex : 1},
          {character : 10, length : 2, scopeIndex : 2},
        ]
      },
      {
        line : 2,
        tokens : [
          {character : 3, length : 2, scopeIndex : 1},
          {character : 6, length : 2, scopeIndex : 1},
          {character : 8, length : 2, scopeIndex : 2},
        ]
      },
    ];

    highlighter.highlight(fileUri1, highlightingsInLine);
    assert.deepEqual(highlighter.applicationUriHistory,
                     [ fileUri1Str, fileUri1Str, fileUri2Str, fileUri1Str ]);
    assert.deepEqual(highlighter.getDecorationRanges(fileUri1),
                     createHighlightingScopeRanges(highlightingsInLine));
    // Keeps state separate between files.
    const highlightingsInLine1:
        semanticHighlighting.SemanticHighlightingLine = {
      line : 1,
      tokens : [
        {character : 2, length : 1, scopeIndex : 0},
      ]
    };
    highlighter.highlight(fileUri2, [ highlightingsInLine1 ]);
    assert.deepEqual(
        highlighter.applicationUriHistory,
        [ fileUri1Str, fileUri1Str, fileUri2Str, fileUri1Str, fileUri2Str ]);
    assert.deepEqual(highlighter.getDecorationRanges(fileUri2),
                     createHighlightingScopeRanges([ highlightingsInLine1 ]));
    // Does full colorizations.
    highlighter.highlight(fileUri1, [ highlightingsInLine1 ]);
    assert.deepEqual(highlighter.applicationUriHistory, [
      fileUri1Str, fileUri1Str, fileUri2Str, fileUri1Str, fileUri2Str,
      fileUri1Str
    ]);
    // After the incremental update to line 1, the old highlightings at line 1
    // will no longer exist in the array.
    assert.deepEqual(
        highlighter.getDecorationRanges(fileUri1),
        createHighlightingScopeRanges(
            [ highlightingsInLine1, ...highlightingsInLine.slice(1) ]));
    // Closing a text document removes all highlightings for the file and no
    // other files.
    highlighter.removeFileHighlightings(fileUri1);
    assert.deepEqual(highlighter.getDecorationRanges(fileUri1), []);
    assert.deepEqual(highlighter.getDecorationRanges(fileUri2),
                     createHighlightingScopeRanges([ highlightingsInLine1 ]));
  });
});
