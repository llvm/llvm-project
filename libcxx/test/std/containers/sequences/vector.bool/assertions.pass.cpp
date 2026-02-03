#include <vector>

#include "test_macros.h"
#include "check_assertion.h"


int main(int, char**) {
#if defined(_LIBCPP_HARDENING_MODE) && _LIBCPP_HARDENING_MODE != 0
    {
        std::vector<bool> v(3);
        TEST_LIBCPP_ASSERT_FAILURE(
            (void)v[3],
            "vector<bool>::operator[] index out of bounds"
        );
    }

    {
        const std::vector<bool> v(3);
        TEST_LIBCPP_ASSERT_FAILURE(
            (void)v[100],
            "vector<bool>::operator[] index out of bounds"
        );
    }
#endif

    return 0;
}
