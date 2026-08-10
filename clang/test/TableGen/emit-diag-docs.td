// RUN: clang-tblgen -gen-diag-docs -I%S %s -o - 2>&1 | \
// RUN:    FileCheck --strict-whitespace %s
include "DiagnosticBase.inc"

def MyGroup : DiagGroup<"MyGroupName">;

def MyKinds : TextSubstitution<"%select{food|forests}0">;
def MyGoodBad : TextSubstitution<"%select{good|bad}0">;
def MySubNested : TextSubstitution<"%sub{MyGoodBad}1 %sub{MyKinds}2 are %sub{MyGoodBad}1 according to %0">;

// CHECK: -WMyGroupName
// CHECK: **Diagnostic text:**

let Group = MyGroup in {

// CHECK:      |:warning:`warning:` |nbsp| |+--------------------------------------------------------------------+|
// CHECK-NEXT: |                           ||+------------------------------------------------------------------+||
// CHECK-NEXT: |                           |||:placeholder:`A` |nbsp| :diagtext:`is not` |nbsp| :placeholder:`B`|||
// CHECK-NEXT: |                           ||+------------------------------------------------------------------+||
// CHECK-NEXT: |                           |+--------------------------------------------------------------------+|
// CHECK-NEXT: |                           ||:diagtext:`this is my diff text`                                    ||
// CHECK-NEXT: |                           |+--------------------------------------------------------------------+|
// CHECK-NEXT: +---------------------------+----------------------------------------------------------------------+
def CheckDiff : Warning<"%diff{$ is not $|this is my diff text}0,1">;


// CHECK:      |:warning:`warning:` |nbsp| :placeholder:`A` |nbsp| :diagtext:`is my modifier test` |nbsp| :placeholder:`B`|
// CHECK-NEXT: +----------------------------------------------------------------------------------------------------------+
def CheckModifier : Warning<"%0 is my modifier test %1">;


// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`This is the` |nbsp| :placeholder:`A` |nbsp| :diagtext:`test I've written`|
// CHECK-NEXT: +---------------------------------------------------------------------------------------------------------------+
def CheckOrdinal : Warning<"This is the %ordinal0 test I've written">;

// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`I wrote` |nbsp| |+----------------+| |nbsp| :diagtext:`tests`|
// CHECK-NEXT: |                                                      ||:diagtext:`no`  ||                         |
// CHECK-NEXT: |                                                      |+----------------+|                         |
// CHECK-NEXT: |                                                      ||:diagtext:`one` ||                         |
// CHECK-NEXT: |                                                      |+----------------+|                         |
// CHECK-NEXT: |                                                      ||:placeholder:`A`||                         |
// CHECK-NEXT: |                                                      |+----------------+|                         |
// CHECK-NEXT: +------------------------------------------------------+------------------+-------------------------+
def CheckPlural : Warning<"I wrote %plural{0:no|1:one|:%0}0 tests">;


// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`bad type` |nbsp| :placeholder:`A`|
// CHECK-NEXT: +-----------------------------------------------------------------------+
def CheckQ : Warning<"bad type %q0">;


// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`My test`|+-------------+| |nbsp| :diagtext:`are the best!`|
// CHECK-NEXT: |                                              ||             ||                                 |
// CHECK-NEXT: |                                              |+-------------+|                                 |
// CHECK-NEXT: |                                              ||:diagtext:`s`||                                 |
// CHECK-NEXT: |                                              |+-------------+|                                 |
// CHECK-NEXT: +----------------------------------------------+---------------+---------------------------------+
def CheckS : Warning<"My test%s0 are the best!">;


// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`this is my select test:` |nbsp| |+---------------+|
// CHECK-NEXT: |                                                                      ||:diagtext:`one`||
// CHECK-NEXT: |                                                                      |+---------------+|
// CHECK-NEXT: |                                                                      ||:diagtext:`two`||
// CHECK-NEXT: |                                                                      |+---------------+|
// CHECK-NEXT: +----------------------------------------------------------------------+-----------------+
def CheckSelect : Warning<"this is my select test: %select{one|two}0 and it is %select{good|bad}1">;


// CHECK:      +-------------------------------------------------------+------------------+--------+---------------------+-------------------------------+------------------+--------------------------------------------------------+
// CHECK-NEXT: |:warning:`warning:` |nbsp| :diagtext:`They say` |nbsp| |+----------------+| |nbsp| |+-------------------+| |nbsp| :diagtext:`are` |nbsp| |+----------------+| |nbsp| :diagtext:`according to` |nbsp| :placeholder:`D`|
// CHECK-NEXT: |                                                       ||:diagtext:`good`||        ||:diagtext:`food`   ||                               ||:diagtext:`good`||                                                        |
// CHECK-NEXT: |                                                       |+----------------+|        |+-------------------+|                               |+----------------+|                                                        |
// CHECK-NEXT: |                                                       ||:diagtext:`bad` ||        ||:diagtext:`forests`||                               ||:diagtext:`bad` ||                                                        |
// CHECK-NEXT: |                                                       |+----------------+|        |+-------------------+|                               |+----------------+|                                                        |
// CHECK-NEXT: +-------------------------------------------------------+------------------+--------+---------------------+-------------------------------+------------------+--------------------------------------------------------+
def CheckSubstitution : Warning<"They say %sub{MySubNested}3,1,0">;


// CHECK:      |:warning:`warning:` |nbsp| :diagtext:`this is my warning text`|
// CHECK-NEXT: +--------------------------------------------------------------+
def CheckText : Warning<"this is my warning text">;

}
