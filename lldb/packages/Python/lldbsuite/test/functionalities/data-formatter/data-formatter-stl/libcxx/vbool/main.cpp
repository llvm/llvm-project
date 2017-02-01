#include <string>
#ifdef _LIBCPP_INLINE_VISIBILITY
#undef _LIBCPP_INLINE_VISIBILITY
#endif
#define _LIBCPP_INLINE_VISIBILITY

#include <vector>

int main()
{
    std::vector<bool> vBool;

    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);

    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);

    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);

    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);

    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(false);
    vBool.push_back(true);
    vBool.push_back(true);

    printf ("size: %d", (int) vBool.size()); // Set break point at this line.
    return 0; 
}
