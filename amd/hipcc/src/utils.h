#ifndef HIP_UTILS_H
#define HIP_UTILS_H

#include <string>
#include <vector>

namespace hipcc {
namespace utils {
// gets the path of the executable name
std::string getSelfPath();

// splits the string with the delimiter
std::vector<std::string> splitStr(std::string const &fullStr, char delimiter);

} // namespace utils
} // namespace hipcc

#endif
