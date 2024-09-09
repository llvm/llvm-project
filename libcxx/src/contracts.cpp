#include <__config>
#include <contracts>
#include <exception>
#include <iostream>




namespace std::contracts {

void invoke_default_contract_violation_handler(const contract_violation& violation) noexcept {
  ::handle_contract_violation(violation);
}

static void display_contract_violation(const contract_violation& violation) noexcept {
  using namespace std::contracts;
  std::cerr << violation.file_name() << ":" << violation.line() << ": ";
  auto assert_str = [&]() -> std::pair<const char*, const char*> {
    switch (violation.kind()) {
    case _AssertKind::pre:
      return {"pre(", ")"};
    case _AssertKind::post:
      return {"post(", ")"};

    case _AssertKind::assert:
      return {"contract_assert(", ")"};

    case _AssertKind::__unknown:
      return {"", ""};
    }
  }();
  std::cerr << assert_str.first << violation.comment() << assert_str.second;
  if (violation.detection_mode() == _DetectionMode::predicate_false) {
    std::cerr << " failed" << std::endl;
  } else {
    std::cerr << " exited via exception" << std::endl;
  }
}

} // namespace std::contracts

struct _BuiltinContractStruct {
  int32_t version;
  int32_t contract_kind;
  int32_t eval_semantic;
  int32_t detection_mode;
  int32_t lineno;
  const char* comment;
  const char* file;
};

extern "C" {
_LIBCPP_EXPORTED_FROM_ABI void __handle_contract_violation(
    unsigned kind,
    unsigned eval_semantic,
    unsigned detection_mode,
    const char* comment,
    const char* file,
    unsigned line) {
  using namespace std::contracts;

  using _InfoT = std::contracts::contract_violation::_Info;
  _InfoT info = {.__kind_ = static_cast<_AssertKind>(kind),
                 .__semantic_ = static_cast<_EvaluationSemantic>(eval_semantic),
                 .__detection_mode_ = static_cast<_DetectionMode>(detection_mode),
                 .__comment_ = comment,
                 .__file_name_ = file,
                 .__function_name_ = nullptr,
                 .__line_ = line};
  contract_violation violation(info);
  if (::handle_contract_violation)
    ::handle_contract_violation(violation);
  else
    std::contracts::display_contract_violation(violation);

  if (info.__semantic_ == _EvaluationSemantic::enforce) {
    std::terminate();
  }
}

_LIBCPP_EXPORTED_FROM_ABI void __handle_contract_violation_new(void* dataptr) noexcept {
  using namespace std::contracts;
  _BuiltinContractStruct* data = static_cast<_BuiltinContractStruct*>(dataptr);

  using _InfoT = std::contracts::contract_violation::_Info;
  _InfoT info  = {
       .__kind_           = static_cast<_AssertKind>(data->contract_kind),
       .__semantic_       = static_cast<_EvaluationSemantic>(data->eval_semantic),
       .__detection_mode_ = static_cast<_DetectionMode>(data->detection_mode),
       .__comment_        = data->comment,
       .__file_name_      = data->file,
       .__function_name_  = nullptr,
       .__line_           = unsigned(data->lineno)};
  contract_violation violation(info);
  if (::handle_contract_violation)
    ::handle_contract_violation(violation);
  else
    std::contracts::display_contract_violation(violation);

  if (info.__semantic_ == _EvaluationSemantic::enforce) {
    std::terminate();
  }
}
}
