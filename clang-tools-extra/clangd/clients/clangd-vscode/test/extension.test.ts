/** The module 'assert' provides assertion methods from node */
import * as assert from 'assert';

import * as vscode from 'vscode';
import * as myExtension from '../src/extension';

// TODO: add tests
suite("Extension Tests", () => {

    // Defines a Mocha unit test
    test("Something 1", () => {
        assert.equal(-1, [1, 2, 3].indexOf(5));
        assert.equal(-1, [1, 2, 3].indexOf(0));
    });
});