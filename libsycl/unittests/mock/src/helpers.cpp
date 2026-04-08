#include "ur_mock_helpers.hpp"

namespace mock {
static callbacks_t callbacks = {};

callbacks_t &getCallbacks() { return callbacks; }

} // namespace mock
