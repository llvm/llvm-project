#include <list>
#include <optional>

#include "helpers.hpp"

namespace unittest {

class OffloadMock {
public:
  OffloadMock() = default;

  OffloadMock(OffloadMock &&Other) = delete;
  OffloadMock(const OffloadMock &) = delete;
  OffloadMock &operator=(const OffloadMock &) = delete;
  ~OffloadMock() {
    // mock::getCallbacks() is an application lifetime object, we need to reset
    // these between tests
    mock::getCallbacks().resetCallbacks();
  }

  template <typename ParamType, typename... Args>
  static ol_result_t callCallback(std::string FunctionName, Args &&...args) {
    auto Callback = mock::getCallbacks().getCallback(FunctionName);
    if (!Callback)
      return mock::getErrorUnimplementedFunction(FunctionName);

    ParamType params = {&args...};
    return Callback(&params);
  }
};

} // namespace unittest

// C++20 std::source_location::function_name
ol_result_t olCreateEvent(ol_queue_handle_t Queue, ol_event_handle_t *Event) {
  return unittest::OffloadMock::callCallback<ol_create_event_params_t>(
      __func__, Queue, Event);
}
