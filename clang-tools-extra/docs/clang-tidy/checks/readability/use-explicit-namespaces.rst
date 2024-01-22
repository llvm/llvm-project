.. title:: clang-tidy - readability-use-explicit-namespaces

readability-use-explicit-namespaces
===================================

This check detects and fixes references to members of namespaces where the namespace is not explicity specified in the reference.  By default, this will only change references that are found through a using namespace directive.

Example:
using namespace std;
using std::string;
string something = "something";
cout << something;


Fixed:
using namespace std;
using std::string;
string something = "something";    // this is not change by default because it is referenced through using std::string instead of using namespace std
std::cout << something;

Options:
LimitToPattern - only change things that match pattern e.g. "value: std" would limit changes to the std namespace
OnlyExpandUsingNamespace - defaults to true meaning that only references found by using namespace are changed
DiagnosticLevel - add diagnostic information about the choices being made by this check and what it is internally using
