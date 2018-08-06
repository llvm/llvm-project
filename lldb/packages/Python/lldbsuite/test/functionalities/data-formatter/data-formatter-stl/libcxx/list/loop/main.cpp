// Evil hack: To simulate memory corruption, we want to fiddle with some internals of std::list.
// Make those accessible to us.
#define private public
#define protected public

#include <list>
#include <stdio.h>
#include <assert.h>

typedef std::list<int> int_list;

int main()
{
#ifdef LLDB_USING_LIBCPP
    int_list *numbers_list = new int_list{1,2,3,4,5,6,7,8,9,10};

    printf("// Set break point at this line.");
    auto *third_elem = numbers_list->__end_.__next_->__next_->__next_;
    assert(third_elem->__value_ == 3);
    auto *fifth_elem = third_elem->__next_->__next_;
    assert(fifth_elem->__value_ == 5);
    fifth_elem->__next_ = third_elem;
#endif

    // Any attempt to free the list will probably crash the program. Let's just leak it.
    return 0; // Set second break point at this line.
}
