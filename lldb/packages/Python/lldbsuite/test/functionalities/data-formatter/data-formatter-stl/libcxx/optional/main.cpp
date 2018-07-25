#include <cstdio>
#include <string>
#include <vector>
#include <optional>

using int_vect = std::vector<int> ;
using optional_int = std::optional<int> ;
using optional_int_vect = std::optional<int_vect> ;
using optional_string = std::optional<std::string> ;

int main()
{
    optional_int number_not_engaged ;
    optional_int number_engaged = 42 ;

    printf( "%d\n", *number_engaged) ;

    optional_int_vect numbers{{1,2,3,4}} ;

    printf( "%d %d\n", numbers.value()[0], numbers.value()[1] ) ;

    optional_string ostring = "hello" ;

    printf( "%s\n", ostring->c_str() ) ;

    return 0; // break here
}
