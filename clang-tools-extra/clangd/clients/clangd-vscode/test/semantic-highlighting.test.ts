import * as assert from 'assert';
import * as path from 'path';

import * as TM from '../src/semantic-highlighting';

suite('TextMate Tests', () => {
  test('Parses arrays of textmate themes.', async () => {
    const themePath =
        path.join(__dirname, '../../test/assets/includeTheme.jsonc');
    const scopeColorRules = await TM.parseThemeFile(themePath);
    const getScopeRule = (scope: string) =>
        scopeColorRules.find((v) => v.scope === scope);
    assert.equal(scopeColorRules.length, 3);
    assert.deepEqual(getScopeRule('a'), {scope : 'a', textColor : '#fff'});
    assert.deepEqual(getScopeRule('b'), {scope : 'b', textColor : '#000'});
    assert.deepEqual(getScopeRule('c'), {scope : 'c', textColor : '#bcd'});
  });
});
