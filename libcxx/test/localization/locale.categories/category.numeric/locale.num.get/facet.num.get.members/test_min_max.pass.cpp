#include <limits>
#include <sstream>
#include <iostream>
#include <cassert>
#include <iostream>

using namespace std;

template<typename T>
void check_limits()
{
    T minv = numeric_limits<T>::min();
    T maxv = numeric_limits<T>::max();

    ostringstream miniss, maxiss;
    assert(miniss << minv);
    assert(maxiss << maxv);
    std::string mins = miniss.str();
    std::string maxs = maxiss.str(); 

    istringstream maxoss(maxs), minoss(mins);

    T new_minv, new_maxv;
    assert(maxoss >> new_maxv);
    assert(minoss >> new_minv);
 
    assert(new_minv == minv);
    assert(new_maxv == maxv);

    if(mins == "0")
        mins = "-1";
    else
        mins[mins.size() - 1]++;
    
    maxs[maxs.size() - 1]++;

    istringstream maxoss2(maxs), minoss2(mins);
    
    assert(! (maxoss2 >> new_maxv));
    assert(! (minoss2 >> new_minv));
}

int main(void)
{
    check_limits<short>();
    check_limits<unsigned short>();
    check_limits<int>();
    check_limits<unsigned int>();
    check_limits<long>();
    check_limits<unsigned long>();
    check_limits<long long>();
    check_limits<unsigned long long>();
}
