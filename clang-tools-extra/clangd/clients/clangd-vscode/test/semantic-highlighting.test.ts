import * as assert from 'assert';
import * as path from 'path';

import * as TM from '../src/semantic-highlighting';

suite('SemanticHighlighting Tests', () => {
  test('Parses arrays of textmate themes.', async () => {
    const themePath =
        path.join(__dirname, '../../test/assets/includeTheme.jsonc');
    const scopeColorRules = await TM.parseThemeFile(themePath);
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
    testCases.forEach((testCase, i) => assert.deepEqual(
                          TM.decodeTokens(testCase), expected[i]));
  });
});
