// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fblocks -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_strongCast' %t  | count 4
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -fblocks -fobjc-gc -emit-llvm -o %t %s
// RUN: grep -F '@objc_assign_strongCast' %t  | count 4

@interface DSATextSearch @end

DSATextSearch **_uniqueIdToIdentifierArray = (0);
void foo (int _nextId)
{
	_uniqueIdToIdentifierArray[_nextId] = 0;  // objc_assign_strongCast
}

typedef struct {
    unsigned long state;
    id *itemsPtr;
    void (^bp)();
    unsigned long *mutationsPtr;
    unsigned long extra[5];
} NSFastEnumerationState;

void foo1 (NSFastEnumerationState * state)
{
   state->itemsPtr = 0;
   state->bp = ^{};
}

