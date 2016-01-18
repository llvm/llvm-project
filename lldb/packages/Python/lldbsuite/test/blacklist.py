"""
'blacklist' is a Python dictionary, it stores the mapping of a string describing
either a testclass or a testcase, i.e, testclass.testmethod, to the reason (a
string) it is blacklisted.

Following is an example which states that test class IntegerTypesExprTestCase
should be skipped because 'This test class crashed' and the test case
FoundationTestCase.test_data_type_and_expr_with_dsym should be skipped because
it is 'Temporarily disabled'.

blacklist = {'IntegerTypesExprTestCase': 'This test class crashed',
             'FoundationTestCase.test_data_type_and_expr_with_dsym': 'Temporarily disabled'
             }
"""

blacklist = {'STLTestCase': '<rdar://problem/8837118> Crashed while running the entire test suite'
             # To skip this test case: ./dotest.py -b blacklist.py -v -w 2> ~/Developer/Log/lldbtest.log
             }
