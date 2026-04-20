#include "helpers.hpp"

namespace mock {
static Callbacks callbacks = {};

Callbacks &getCallbacks() { return callbacks; }

} // namespace mock
