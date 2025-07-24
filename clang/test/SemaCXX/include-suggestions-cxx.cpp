// RUN: %clang_cc1 -verify -fsyntax-only %s 

namespace std{};

void test(){
  std::FILE; // expected-error {{no member}} expected-note {{'std::FILE' is defined in}}
  std::_Exit; // expected-error {{no member}} expected-note {{'std::_Exit' is defined in}} expected-note {{'std::_Exit' is a}}
  std::accumulate; // expected-error {{no member}} expected-note {{'std::accumulate' is defined in}}
  std::acosf; // expected-error {{no member}} expected-note {{'std::acosf' is defined in}} expected-note {{'std::acosf' is a}}
  std::acoshf; // expected-error {{no member}} expected-note {{'std::acoshf' is defined in}} expected-note {{'std::acoshf' is a}}
  std::acoshl; // expected-error {{no member}} expected-note {{'std::acoshl' is defined in}} expected-note {{'std::acoshl' is a}}
  std::acosl; // expected-error {{no member}} expected-note {{'std::acosl' is defined in}} expected-note {{'std::acosl' is a}}
  std::add_const; // expected-error {{no member}} expected-note {{'std::add_const' is defined in}} expected-note {{'std::add_const' is a}}
  std::add_const_t; // expected-error {{no member}} expected-note {{'std::add_const_t' is defined in}} expected-note {{'std::add_const_t' is a}}
  std::add_cv; // expected-error {{no member}} expected-note {{'std::add_cv' is defined in}} expected-note {{'std::add_cv' is a}}
  std::add_cv_t; // expected-error {{no member}} expected-note {{'std::add_cv_t' is defined in}} expected-note {{'std::add_cv_t' is a}}
  std::add_lvalue_reference; // expected-error {{no member}} expected-note {{'std::add_lvalue_reference' is defined in}} expected-note {{'std::add_lvalue_reference' is a}}
  std::add_lvalue_reference_t; // expected-error {{no member}} expected-note {{'std::add_lvalue_reference_t' is defined in}} expected-note {{'std::add_lvalue_reference_t' is a}}
  std::add_pointer; // expected-error {{no member}} expected-note {{'std::add_pointer' is defined in}} expected-note {{'std::add_pointer' is a}}
  std::add_pointer_t; // expected-error {{no member}} expected-note {{'std::add_pointer_t' is defined in}} expected-note {{'std::add_pointer_t' is a}}
  std::add_rvalue_reference; // expected-error {{no member}} expected-note {{'std::add_rvalue_reference' is defined in}} expected-note {{'std::add_rvalue_reference' is a}}
  std::add_rvalue_reference_t; // expected-error {{no member}} expected-note {{'std::add_rvalue_reference_t' is defined in}} expected-note {{'std::add_rvalue_reference_t' is a}}
  std::add_sat; // expected-error {{no member}} expected-note {{'std::add_sat' is defined in}} expected-note {{'std::add_sat' is a}}
  std::add_volatile; // expected-error {{no member}} expected-note {{'std::add_volatile' is defined in}} expected-note {{'std::add_volatile' is a}}
  std::add_volatile_t; // expected-error {{no member}} expected-note {{'std::add_volatile_t' is defined in}} expected-note {{'std::add_volatile_t' is a}}
  std::addressof; // expected-error {{no member}} expected-note {{'std::addressof' is defined in}} expected-note {{'std::addressof' is a}}
  std::adjacent_difference; // expected-error {{no member}} expected-note {{'std::adjacent_difference' is defined in}}
  std::adjacent_find; // expected-error {{no member}} expected-note {{'std::adjacent_find' is defined in}}
  std::adopt_lock; // expected-error {{no member}} expected-note {{'std::adopt_lock' is defined in}} expected-note {{'std::adopt_lock' is a}}
  std::adopt_lock_t; // expected-error {{no member}} expected-note {{'std::adopt_lock_t' is defined in}} expected-note {{'std::adopt_lock_t' is a}}
  std::advance; // expected-error {{no member}} expected-note {{'std::advance' is defined in}}
  std::align; // expected-error {{no member}} expected-note {{'std::align' is defined in}} expected-note {{'std::align' is a}}
  std::align_val_t; // expected-error {{no member}} expected-note {{'std::align_val_t' is defined in}} expected-note {{'std::align_val_t' is a}}
  std::aligned_alloc; // expected-error {{no member}} expected-note {{'std::aligned_alloc' is defined in}} expected-note {{'std::aligned_alloc' is a}}
  std::aligned_storage; // expected-error {{no member}} expected-note {{'std::aligned_storage' is defined in}} expected-note {{'std::aligned_storage' is a}}
  std::aligned_storage_t; // expected-error {{no member}} expected-note {{'std::aligned_storage_t' is defined in}} expected-note {{'std::aligned_storage_t' is a}}
  std::aligned_union; // expected-error {{no member}} expected-note {{'std::aligned_union' is defined in}} expected-note {{'std::aligned_union' is a}}
  std::aligned_union_t; // expected-error {{no member}} expected-note {{'std::aligned_union_t' is defined in}} expected-note {{'std::aligned_union_t' is a}}
  std::alignment_of; // expected-error {{no member}} expected-note {{'std::alignment_of' is defined in}} expected-note {{'std::alignment_of' is a}}
  std::alignment_of_v; // expected-error {{no member}} expected-note {{'std::alignment_of_v' is defined in}} expected-note {{'std::alignment_of_v' is a}}
  std::all_of; // expected-error {{no member}} expected-note {{'std::all_of' is defined in}} expected-note {{'std::all_of' is a}}
  std::allocate_shared; // expected-error {{no member}} expected-note {{'std::allocate_shared' is defined in}} expected-note {{'std::allocate_shared' is a}}
  std::allocate_shared_for_overwrite; // expected-error {{no member}} expected-note {{'std::allocate_shared_for_overwrite' is defined in}} expected-note {{'std::allocate_shared_for_overwrite' is a}}
  std::allocation_result; // expected-error {{no member}} expected-note {{'std::allocation_result' is defined in}} expected-note {{'std::allocation_result' is a}}
  std::allocator; // expected-error {{no member}} expected-note {{'std::allocator' is defined in}}
  std::allocator_arg; // expected-error {{no member}} expected-note {{'std::allocator_arg' is defined in}} expected-note {{'std::allocator_arg' is a}}
  std::allocator_arg_t; // expected-error {{no member}} expected-note {{'std::allocator_arg_t' is defined in}} expected-note {{'std::allocator_arg_t' is a}}
  std::allocator_traits; // expected-error {{no member}} expected-note {{'std::allocator_traits' is defined in}} expected-note {{'std::allocator_traits' is a}}
  std::any; // expected-error {{no member}} expected-note {{'std::any' is defined in}} expected-note {{'std::any' is a}}
  std::any_cast; // expected-error {{no member}} expected-note {{'std::any_cast' is defined in}} expected-note {{'std::any_cast' is a}}
  std::any_of; // expected-error {{no member}} expected-note {{'std::any_of' is defined in}} expected-note {{'std::any_of' is a}}
  std::apply; // expected-error {{no member}} expected-note {{'std::apply' is defined in}} expected-note {{'std::apply' is a}}
  std::arg; // expected-error {{no member}} expected-note {{'std::arg' is defined in}}
  std::array; // expected-error {{no member}} expected-note {{'std::array' is defined in}} expected-note {{'std::array' is a}}
  std::as_bytes; // expected-error {{no member}} expected-note {{'std::as_bytes' is defined in}} expected-note {{'std::as_bytes' is a}}
  std::as_const; // expected-error {{no member}} expected-note {{'std::as_const' is defined in}} expected-note {{'std::as_const' is a}}
  std::as_writable_bytes; // expected-error {{no member}} expected-note {{'std::as_writable_bytes' is defined in}} expected-note {{'std::as_writable_bytes' is a}}
  std::asctime; // expected-error {{no member}} expected-note {{'std::asctime' is defined in}}
  std::asinf; // expected-error {{no member}} expected-note {{'std::asinf' is defined in}} expected-note {{'std::asinf' is a}}
  std::asinhf; // expected-error {{no member}} expected-note {{'std::asinhf' is defined in}} expected-note {{'std::asinhf' is a}}
  std::asinhl; // expected-error {{no member}} expected-note {{'std::asinhl' is defined in}} expected-note {{'std::asinhl' is a}}
  std::asinl; // expected-error {{no member}} expected-note {{'std::asinl' is defined in}} expected-note {{'std::asinl' is a}}
  std::assignable_from; // expected-error {{no member}} expected-note {{'std::assignable_from' is defined in}} expected-note {{'std::assignable_from' is a}}
  std::assoc_laguerre; // expected-error {{no member}} expected-note {{'std::assoc_laguerre' is defined in}} expected-note {{'std::assoc_laguerre' is a}}
  std::assoc_laguerref; // expected-error {{no member}} expected-note {{'std::assoc_laguerref' is defined in}} expected-note {{'std::assoc_laguerref' is a}}
  std::assoc_laguerrel; // expected-error {{no member}} expected-note {{'std::assoc_laguerrel' is defined in}} expected-note {{'std::assoc_laguerrel' is a}}
  std::assoc_legendre; // expected-error {{no member}} expected-note {{'std::assoc_legendre' is defined in}} expected-note {{'std::assoc_legendre' is a}}
  std::assoc_legendref; // expected-error {{no member}} expected-note {{'std::assoc_legendref' is defined in}} expected-note {{'std::assoc_legendref' is a}}
  std::assoc_legendrel; // expected-error {{no member}} expected-note {{'std::assoc_legendrel' is defined in}} expected-note {{'std::assoc_legendrel' is a}}
  std::assume_aligned; // expected-error {{no member}} expected-note {{'std::assume_aligned' is defined in}} expected-note {{'std::assume_aligned' is a}}
  std::async; // expected-error {{no member}} expected-note {{'std::async' is defined in}} expected-note {{'std::async' is a}}
  std::at_quick_exit; // expected-error {{no member}} expected-note {{'std::at_quick_exit' is defined in}} expected-note {{'std::at_quick_exit' is a}}
  std::atan2f; // expected-error {{no member}} expected-note {{'std::atan2f' is defined in}} expected-note {{'std::atan2f' is a}}
  std::atan2l; // expected-error {{no member}} expected-note {{'std::atan2l' is defined in}} expected-note {{'std::atan2l' is a}}
  std::atanf; // expected-error {{no member}} expected-note {{'std::atanf' is defined in}} expected-note {{'std::atanf' is a}}
  std::atanhf; // expected-error {{no member}} expected-note {{'std::atanhf' is defined in}} expected-note {{'std::atanhf' is a}}
  std::atanhl; // expected-error {{no member}} expected-note {{'std::atanhl' is defined in}} expected-note {{'std::atanhl' is a}}
  std::atanl; // expected-error {{no member}} expected-note {{'std::atanl' is defined in}} expected-note {{'std::atanl' is a}}
  std::atexit; // expected-error {{no member}} expected-note {{'std::atexit' is defined in}}
  std::atof; // expected-error {{no member}} expected-note {{'std::atof' is defined in}}
  std::atoi; // expected-error {{no member}} expected-note {{'std::atoi' is defined in}}
  std::atol; // expected-error {{no member}} expected-note {{'std::atol' is defined in}}
  std::atoll; // expected-error {{no member}} expected-note {{'std::atoll' is defined in}} expected-note {{'std::atoll' is a}}
  std::atomic_compare_exchange_strong; // expected-error {{no member}} expected-note {{'std::atomic_compare_exchange_strong' is defined in}} expected-note {{'std::atomic_compare_exchange_strong' is a}}
  std::atomic_compare_exchange_strong_explicit; // expected-error {{no member}} expected-note {{'std::atomic_compare_exchange_strong_explicit' is defined in}} expected-note {{'std::atomic_compare_exchange_strong_explicit' is a}}
  std::atomic_compare_exchange_weak; // expected-error {{no member}} expected-note {{'std::atomic_compare_exchange_weak' is defined in}} expected-note {{'std::atomic_compare_exchange_weak' is a}}
  std::atomic_compare_exchange_weak_explicit; // expected-error {{no member}} expected-note {{'std::atomic_compare_exchange_weak_explicit' is defined in}} expected-note {{'std::atomic_compare_exchange_weak_explicit' is a}}
  std::atomic_exchange; // expected-error {{no member}} expected-note {{'std::atomic_exchange' is defined in}} expected-note {{'std::atomic_exchange' is a}}
  std::atomic_exchange_explicit; // expected-error {{no member}} expected-note {{'std::atomic_exchange_explicit' is defined in}} expected-note {{'std::atomic_exchange_explicit' is a}}
  std::atomic_fetch_add; // expected-error {{no member}} expected-note {{'std::atomic_fetch_add' is defined in}} expected-note {{'std::atomic_fetch_add' is a}}
  std::atomic_fetch_add_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_add_explicit' is defined in}} expected-note {{'std::atomic_fetch_add_explicit' is a}}
  std::atomic_fetch_and; // expected-error {{no member}} expected-note {{'std::atomic_fetch_and' is defined in}} expected-note {{'std::atomic_fetch_and' is a}}
  std::atomic_fetch_and_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_and_explicit' is defined in}} expected-note {{'std::atomic_fetch_and_explicit' is a}}
  std::atomic_fetch_max; // expected-error {{no member}} expected-note {{'std::atomic_fetch_max' is defined in}} expected-note {{'std::atomic_fetch_max' is a}}
  std::atomic_fetch_max_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_max_explicit' is defined in}} expected-note {{'std::atomic_fetch_max_explicit' is a}}
  std::atomic_fetch_min; // expected-error {{no member}} expected-note {{'std::atomic_fetch_min' is defined in}} expected-note {{'std::atomic_fetch_min' is a}}
  std::atomic_fetch_min_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_min_explicit' is defined in}} expected-note {{'std::atomic_fetch_min_explicit' is a}}
  std::atomic_fetch_or; // expected-error {{no member}} expected-note {{'std::atomic_fetch_or' is defined in}} expected-note {{'std::atomic_fetch_or' is a}}
  std::atomic_fetch_or_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_or_explicit' is defined in}} expected-note {{'std::atomic_fetch_or_explicit' is a}}
  std::atomic_fetch_sub; // expected-error {{no member}} expected-note {{'std::atomic_fetch_sub' is defined in}} expected-note {{'std::atomic_fetch_sub' is a}}
  std::atomic_fetch_sub_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_sub_explicit' is defined in}} expected-note {{'std::atomic_fetch_sub_explicit' is a}}
  std::atomic_fetch_xor; // expected-error {{no member}} expected-note {{'std::atomic_fetch_xor' is defined in}} expected-note {{'std::atomic_fetch_xor' is a}}
  std::atomic_fetch_xor_explicit; // expected-error {{no member}} expected-note {{'std::atomic_fetch_xor_explicit' is defined in}} expected-note {{'std::atomic_fetch_xor_explicit' is a}}
  std::atomic_flag; // expected-error {{no member}} expected-note {{'std::atomic_flag' is defined in}} expected-note {{'std::atomic_flag' is a}}
  std::atomic_flag_clear; // expected-error {{no member}} expected-note {{'std::atomic_flag_clear' is defined in}} expected-note {{'std::atomic_flag_clear' is a}}
  std::atomic_flag_clear_explicit; // expected-error {{no member}} expected-note {{'std::atomic_flag_clear_explicit' is defined in}} expected-note {{'std::atomic_flag_clear_explicit' is a}}
  std::atomic_flag_notify_all; // expected-error {{no member}} expected-note {{'std::atomic_flag_notify_all' is defined in}} expected-note {{'std::atomic_flag_notify_all' is a}}
  std::atomic_flag_notify_one; // expected-error {{no member}} expected-note {{'std::atomic_flag_notify_one' is defined in}} expected-note {{'std::atomic_flag_notify_one' is a}}
  std::atomic_flag_test; // expected-error {{no member}} expected-note {{'std::atomic_flag_test' is defined in}} expected-note {{'std::atomic_flag_test' is a}}
  std::atomic_flag_test_and_set; // expected-error {{no member}} expected-note {{'std::atomic_flag_test_and_set' is defined in}} expected-note {{'std::atomic_flag_test_and_set' is a}}
  std::atomic_flag_test_and_set_explicit; // expected-error {{no member}} expected-note {{'std::atomic_flag_test_and_set_explicit' is defined in}} expected-note {{'std::atomic_flag_test_and_set_explicit' is a}}
  std::atomic_flag_test_explicit; // expected-error {{no member}} expected-note {{'std::atomic_flag_test_explicit' is defined in}} expected-note {{'std::atomic_flag_test_explicit' is a}}
  std::atomic_flag_wait; // expected-error {{no member}} expected-note {{'std::atomic_flag_wait' is defined in}} expected-note {{'std::atomic_flag_wait' is a}}
  std::atomic_flag_wait_explicit; // expected-error {{no member}} expected-note {{'std::atomic_flag_wait_explicit' is defined in}} expected-note {{'std::atomic_flag_wait_explicit' is a}}
  std::atomic_init; // expected-error {{no member}} expected-note {{'std::atomic_init' is defined in}} expected-note {{'std::atomic_init' is a}}
  std::atomic_is_lock_free; // expected-error {{no member}} expected-note {{'std::atomic_is_lock_free' is defined in}} expected-note {{'std::atomic_is_lock_free' is a}}
  std::atomic_load; // expected-error {{no member}} expected-note {{'std::atomic_load' is defined in}} expected-note {{'std::atomic_load' is a}}
  std::atomic_load_explicit; // expected-error {{no member}} expected-note {{'std::atomic_load_explicit' is defined in}} expected-note {{'std::atomic_load_explicit' is a}}
  std::atomic_notify_all; // expected-error {{no member}} expected-note {{'std::atomic_notify_all' is defined in}} expected-note {{'std::atomic_notify_all' is a}}
  std::atomic_notify_one; // expected-error {{no member}} expected-note {{'std::atomic_notify_one' is defined in}} expected-note {{'std::atomic_notify_one' is a}}
  std::atomic_ref; // expected-error {{no member}} expected-note {{'std::atomic_ref' is defined in}} expected-note {{'std::atomic_ref' is a}}
  std::atomic_signal_fence; // expected-error {{no member}} expected-note {{'std::atomic_signal_fence' is defined in}} expected-note {{'std::atomic_signal_fence' is a}}
  std::atomic_store; // expected-error {{no member}} expected-note {{'std::atomic_store' is defined in}} expected-note {{'std::atomic_store' is a}}
  std::atomic_store_explicit; // expected-error {{no member}} expected-note {{'std::atomic_store_explicit' is defined in}} expected-note {{'std::atomic_store_explicit' is a}}
  std::atomic_thread_fence; // expected-error {{no member}} expected-note {{'std::atomic_thread_fence' is defined in}} expected-note {{'std::atomic_thread_fence' is a}}
  std::atomic_wait; // expected-error {{no member}} expected-note {{'std::atomic_wait' is defined in}} expected-note {{'std::atomic_wait' is a}}
  std::atomic_wait_explicit; // expected-error {{no member}} expected-note {{'std::atomic_wait_explicit' is defined in}} expected-note {{'std::atomic_wait_explicit' is a}}
  std::atto; // expected-error {{no member}} expected-note {{'std::atto' is defined in}} expected-note {{'std::atto' is a}}
  std::auto_ptr; // expected-error {{no member}} expected-note {{'std::auto_ptr' is defined in}}
  std::back_insert_iterator; // expected-error {{no member}} expected-note {{'std::back_insert_iterator' is defined in}}
  std::back_inserter; // expected-error {{no member}} expected-note {{'std::back_inserter' is defined in}}
  std::bad_alloc; // expected-error {{no member}} expected-note {{'std::bad_alloc' is defined in}}
  std::bad_any_cast; // expected-error {{no member}} expected-note {{'std::bad_any_cast' is defined in}} expected-note {{'std::bad_any_cast' is a}}
  std::bad_array_new_length; // expected-error {{no member}} expected-note {{'std::bad_array_new_length' is defined in}} expected-note {{'std::bad_array_new_length' is a}}
  std::bad_cast; // expected-error {{no member}} expected-note {{'std::bad_cast' is defined in}}
  std::bad_exception; // expected-error {{no member}} expected-note {{'std::bad_exception' is defined in}}
  std::bad_expected_access; // expected-error {{no member}} expected-note {{'std::bad_expected_access' is defined in}} expected-note {{'std::bad_expected_access' is a}}
  std::bad_function_call; // expected-error {{no member}} expected-note {{'std::bad_function_call' is defined in}} expected-note {{'std::bad_function_call' is a}}
  std::bad_optional_access; // expected-error {{no member}} expected-note {{'std::bad_optional_access' is defined in}} expected-note {{'std::bad_optional_access' is a}}
  std::bad_typeid; // expected-error {{no member}} expected-note {{'std::bad_typeid' is defined in}}
  std::bad_variant_access; // expected-error {{no member}} expected-note {{'std::bad_variant_access' is defined in}} expected-note {{'std::bad_variant_access' is a}}
  std::bad_weak_ptr; // expected-error {{no member}} expected-note {{'std::bad_weak_ptr' is defined in}} expected-note {{'std::bad_weak_ptr' is a}}
  std::barrier; // expected-error {{no member}} expected-note {{'std::barrier' is defined in}} expected-note {{'std::barrier' is a}}
  std::basic_common_reference; // expected-error {{no member}} expected-note {{'std::basic_common_reference' is defined in}} expected-note {{'std::basic_common_reference' is a}}
  std::basic_const_iterator; // expected-error {{no member}} expected-note {{'std::basic_const_iterator' is defined in}} expected-note {{'std::basic_const_iterator' is a}}
  std::basic_filebuf; // expected-error {{no member}} expected-note {{'std::basic_filebuf' is defined in}}
  std::basic_filebuf; // expected-error {{no member}} expected-note {{'std::basic_filebuf' is defined in}}
  std::basic_format_arg; // expected-error {{no member}} expected-note {{'std::basic_format_arg' is defined in}} expected-note {{'std::basic_format_arg' is a}}
  std::basic_format_args; // expected-error {{no member}} expected-note {{'std::basic_format_args' is defined in}} expected-note {{'std::basic_format_args' is a}}
  std::basic_format_context; // expected-error {{no member}} expected-note {{'std::basic_format_context' is defined in}} expected-note {{'std::basic_format_context' is a}}
  std::basic_format_parse_context; // expected-error {{no member}} expected-note {{'std::basic_format_parse_context' is defined in}} expected-note {{'std::basic_format_parse_context' is a}}
  std::basic_format_string; // expected-error {{no member}} expected-note {{'std::basic_format_string' is defined in}} expected-note {{'std::basic_format_string' is a}}
  std::basic_fstream; // expected-error {{no member}} expected-note {{'std::basic_fstream' is defined in}}
  std::basic_fstream; // expected-error {{no member}} expected-note {{'std::basic_fstream' is defined in}}
  std::basic_ifstream; // expected-error {{no member}} expected-note {{'std::basic_ifstream' is defined in}}
  std::basic_ifstream; // expected-error {{no member}} expected-note {{'std::basic_ifstream' is defined in}}
  std::basic_ios; // expected-error {{no member}} expected-note {{'std::basic_ios' is defined in}}
  std::basic_ios; // expected-error {{no member}} expected-note {{'std::basic_ios' is defined in}}
  std::basic_ios; // expected-error {{no member}} expected-note {{'std::basic_ios' is defined in}}
  std::basic_iostream; // expected-error {{no member}} expected-note {{'std::basic_iostream' is defined in}}
  std::basic_iostream; // expected-error {{no member}} expected-note {{'std::basic_iostream' is defined in}}
  std::basic_iostream; // expected-error {{no member}} expected-note {{'std::basic_iostream' is defined in}}
  std::basic_ispanstream; // expected-error {{no member}} expected-note {{'std::basic_ispanstream' is defined in}} expected-note {{'std::basic_ispanstream' is a}}
  std::basic_ispanstream; // expected-error {{no member}} expected-note {{'std::basic_ispanstream' is defined in}} expected-note {{'std::basic_ispanstream' is a}}
  std::basic_istream; // expected-error {{no member}} expected-note {{'std::basic_istream' is defined in}}
  std::basic_istream; // expected-error {{no member}} expected-note {{'std::basic_istream' is defined in}}
  std::basic_istream; // expected-error {{no member}} expected-note {{'std::basic_istream' is defined in}}
  std::basic_istringstream; // expected-error {{no member}} expected-note {{'std::basic_istringstream' is defined in}}
  std::basic_istringstream; // expected-error {{no member}} expected-note {{'std::basic_istringstream' is defined in}}
  std::basic_ofstream; // expected-error {{no member}} expected-note {{'std::basic_ofstream' is defined in}}
  std::basic_ofstream; // expected-error {{no member}} expected-note {{'std::basic_ofstream' is defined in}}
  std::basic_ospanstream; // expected-error {{no member}} expected-note {{'std::basic_ospanstream' is defined in}} expected-note {{'std::basic_ospanstream' is a}}
  std::basic_ospanstream; // expected-error {{no member}} expected-note {{'std::basic_ospanstream' is defined in}} expected-note {{'std::basic_ospanstream' is a}}
  std::basic_ostream; // expected-error {{no member}} expected-note {{'std::basic_ostream' is defined in}}
  std::basic_ostream; // expected-error {{no member}} expected-note {{'std::basic_ostream' is defined in}}
  std::basic_ostream; // expected-error {{no member}} expected-note {{'std::basic_ostream' is defined in}}
  std::basic_ostringstream; // expected-error {{no member}} expected-note {{'std::basic_ostringstream' is defined in}}
  std::basic_ostringstream; // expected-error {{no member}} expected-note {{'std::basic_ostringstream' is defined in}}
  std::basic_osyncstream; // expected-error {{no member}} expected-note {{'std::basic_osyncstream' is defined in}} expected-note {{'std::basic_osyncstream' is a}}
  std::basic_osyncstream; // expected-error {{no member}} expected-note {{'std::basic_osyncstream' is defined in}} expected-note {{'std::basic_osyncstream' is a}}
  std::basic_regex; // expected-error {{no member}} expected-note {{'std::basic_regex' is defined in}} expected-note {{'std::basic_regex' is a}}
  std::basic_spanbuf; // expected-error {{no member}} expected-note {{'std::basic_spanbuf' is defined in}} expected-note {{'std::basic_spanbuf' is a}}
  std::basic_spanbuf; // expected-error {{no member}} expected-note {{'std::basic_spanbuf' is defined in}} expected-note {{'std::basic_spanbuf' is a}}
  std::basic_spanstream; // expected-error {{no member}} expected-note {{'std::basic_spanstream' is defined in}} expected-note {{'std::basic_spanstream' is a}}
  std::basic_spanstream; // expected-error {{no member}} expected-note {{'std::basic_spanstream' is defined in}} expected-note {{'std::basic_spanstream' is a}}
  std::basic_stacktrace; // expected-error {{no member}} expected-note {{'std::basic_stacktrace' is defined in}} expected-note {{'std::basic_stacktrace' is a}}
  std::basic_streambuf; // expected-error {{no member}} expected-note {{'std::basic_streambuf' is defined in}}
  std::basic_streambuf; // expected-error {{no member}} expected-note {{'std::basic_streambuf' is defined in}}
  std::basic_streambuf; // expected-error {{no member}} expected-note {{'std::basic_streambuf' is defined in}}
  std::basic_string; // expected-error {{no member}} expected-note {{'std::basic_string' is defined in}}
  std::basic_string_view; // expected-error {{no member}} expected-note {{'std::basic_string_view' is defined in}} expected-note {{'std::basic_string_view' is a}}
  std::basic_stringbuf; // expected-error {{no member}} expected-note {{'std::basic_stringbuf' is defined in}}
  std::basic_stringbuf; // expected-error {{no member}} expected-note {{'std::basic_stringbuf' is defined in}}
  std::basic_stringstream; // expected-error {{no member}} expected-note {{'std::basic_stringstream' is defined in}}
  std::basic_stringstream; // expected-error {{no member}} expected-note {{'std::basic_stringstream' is defined in}}
  std::basic_syncbuf; // expected-error {{no member}} expected-note {{'std::basic_syncbuf' is defined in}} expected-note {{'std::basic_syncbuf' is a}}
  std::basic_syncbuf; // expected-error {{no member}} expected-note {{'std::basic_syncbuf' is defined in}} expected-note {{'std::basic_syncbuf' is a}}
  std::bernoulli_distribution; // expected-error {{no member}} expected-note {{'std::bernoulli_distribution' is defined in}} expected-note {{'std::bernoulli_distribution' is a}}
  std::beta; // expected-error {{no member}} expected-note {{'std::beta' is defined in}} expected-note {{'std::beta' is a}}
  std::betaf; // expected-error {{no member}} expected-note {{'std::betaf' is defined in}} expected-note {{'std::betaf' is a}}
  std::betal; // expected-error {{no member}} expected-note {{'std::betal' is defined in}} expected-note {{'std::betal' is a}}
  std::bidirectional_iterator; // expected-error {{no member}} expected-note {{'std::bidirectional_iterator' is defined in}} expected-note {{'std::bidirectional_iterator' is a}}
  std::bidirectional_iterator_tag; // expected-error {{no member}} expected-note {{'std::bidirectional_iterator_tag' is defined in}}
  std::binary_function; // expected-error {{no member}} expected-note {{'std::binary_function' is defined in}}
  std::binary_negate; // expected-error {{no member}} expected-note {{'std::binary_negate' is defined in}}
  std::binary_search; // expected-error {{no member}} expected-note {{'std::binary_search' is defined in}}
  std::binary_semaphore; // expected-error {{no member}} expected-note {{'std::binary_semaphore' is defined in}} expected-note {{'std::binary_semaphore' is a}}
  std::bind; // expected-error {{no member}} expected-note {{'std::bind' is defined in}} expected-note {{'std::bind' is a}}
  std::bind1st; // expected-error {{no member}} expected-note {{'std::bind1st' is defined in}}
  std::bind2nd; // expected-error {{no member}} expected-note {{'std::bind2nd' is defined in}}
  std::bind_back; // expected-error {{no member}} expected-note {{'std::bind_back' is defined in}} expected-note {{'std::bind_back' is a}}
  std::bind_front; // expected-error {{no member}} expected-note {{'std::bind_front' is defined in}} expected-note {{'std::bind_front' is a}}
  std::binder1st; // expected-error {{no member}} expected-note {{'std::binder1st' is defined in}}
  std::binder2nd; // expected-error {{no member}} expected-note {{'std::binder2nd' is defined in}}
  std::binomial_distribution; // expected-error {{no member}} expected-note {{'std::binomial_distribution' is defined in}} expected-note {{'std::binomial_distribution' is a}}
  std::bit_and; // expected-error {{no member}} expected-note {{'std::bit_and' is defined in}}
  std::bit_cast; // expected-error {{no member}} expected-note {{'std::bit_cast' is defined in}} expected-note {{'std::bit_cast' is a}}
  std::bit_ceil; // expected-error {{no member}} expected-note {{'std::bit_ceil' is defined in}} expected-note {{'std::bit_ceil' is a}}
  std::bit_floor; // expected-error {{no member}} expected-note {{'std::bit_floor' is defined in}} expected-note {{'std::bit_floor' is a}}
  std::bit_not; // expected-error {{no member}} expected-note {{'std::bit_not' is defined in}} expected-note {{'std::bit_not' is a}}
  std::bit_or; // expected-error {{no member}} expected-note {{'std::bit_or' is defined in}}
  std::bit_width; // expected-error {{no member}} expected-note {{'std::bit_width' is defined in}} expected-note {{'std::bit_width' is a}}
  std::bit_xor; // expected-error {{no member}} expected-note {{'std::bit_xor' is defined in}}
  std::bitset; // expected-error {{no member}} expected-note {{'std::bitset' is defined in}}
  std::bool_constant; // expected-error {{no member}} expected-note {{'std::bool_constant' is defined in}} expected-note {{'std::bool_constant' is a}}
  std::boolalpha; // expected-error {{no member}} expected-note {{'std::boolalpha' is defined in}}
  std::boolalpha; // expected-error {{no member}} expected-note {{'std::boolalpha' is defined in}}
  std::boyer_moore_horspool_searcher; // expected-error {{no member}} expected-note {{'std::boyer_moore_horspool_searcher' is defined in}} expected-note {{'std::boyer_moore_horspool_searcher' is a}}
  std::boyer_moore_searcher; // expected-error {{no member}} expected-note {{'std::boyer_moore_searcher' is defined in}} expected-note {{'std::boyer_moore_searcher' is a}}
  std::breakpoint; // expected-error {{no member}} expected-note {{'std::breakpoint' is defined in}} expected-note {{'std::breakpoint' is a}}
  std::breakpoint_if_debugging; // expected-error {{no member}} expected-note {{'std::breakpoint_if_debugging' is defined in}} expected-note {{'std::breakpoint_if_debugging' is a}}
  std::bsearch; // expected-error {{no member}} expected-note {{'std::bsearch' is defined in}}
  std::btowc; // expected-error {{no member}} expected-note {{'std::btowc' is defined in}}
  std::byte; // expected-error {{no member}} expected-note {{'std::byte' is defined in}} expected-note {{'std::byte' is a}}
  std::byteswap; // expected-error {{no member}} expected-note {{'std::byteswap' is defined in}} expected-note {{'std::byteswap' is a}}
  std::c16rtomb; // expected-error {{no member}} expected-note {{'std::c16rtomb' is defined in}} expected-note {{'std::c16rtomb' is a}}
  std::c32rtomb; // expected-error {{no member}} expected-note {{'std::c32rtomb' is defined in}} expected-note {{'std::c32rtomb' is a}}
  std::c8rtomb; // expected-error {{no member}} expected-note {{'std::c8rtomb' is defined in}} expected-note {{'std::c8rtomb' is a}}
  std::call_once; // expected-error {{no member}} expected-note {{'std::call_once' is defined in}} expected-note {{'std::call_once' is a}}
  std::calloc; // expected-error {{no member}} expected-note {{'std::calloc' is defined in}}
  std::cauchy_distribution; // expected-error {{no member}} expected-note {{'std::cauchy_distribution' is defined in}} expected-note {{'std::cauchy_distribution' is a}}
  std::cbrtf; // expected-error {{no member}} expected-note {{'std::cbrtf' is defined in}} expected-note {{'std::cbrtf' is a}}
  std::cbrtl; // expected-error {{no member}} expected-note {{'std::cbrtl' is defined in}} expected-note {{'std::cbrtl' is a}}
  std::ceilf; // expected-error {{no member}} expected-note {{'std::ceilf' is defined in}} expected-note {{'std::ceilf' is a}}
  std::ceill; // expected-error {{no member}} expected-note {{'std::ceill' is defined in}} expected-note {{'std::ceill' is a}}
  std::centi; // expected-error {{no member}} expected-note {{'std::centi' is defined in}} expected-note {{'std::centi' is a}}
  std::cerr; // expected-error {{no member}} expected-note {{'std::cerr' is defined in}}
  std::char_traits; // expected-error {{no member}} expected-note {{'std::char_traits' is defined in}}
  std::chars_format; // expected-error {{no member}} expected-note {{'std::chars_format' is defined in}} expected-note {{'std::chars_format' is a}}
  std::chi_squared_distribution; // expected-error {{no member}} expected-note {{'std::chi_squared_distribution' is defined in}} expected-note {{'std::chi_squared_distribution' is a}}
  std::cin; // expected-error {{no member}} expected-note {{'std::cin' is defined in}}
  std::clamp; // expected-error {{no member}} expected-note {{'std::clamp' is defined in}} expected-note {{'std::clamp' is a}}
  std::clearerr; // expected-error {{no member}} expected-note {{'std::clearerr' is defined in}}
  std::clock; // expected-error {{no member}} expected-note {{'std::clock' is defined in}}
  std::clock_t; // expected-error {{no member}} expected-note {{'std::clock_t' is defined in}}
  std::clog; // expected-error {{no member}} expected-note {{'std::clog' is defined in}}
  std::cmatch; // expected-error {{no member}} expected-note {{'std::cmatch' is defined in}} expected-note {{'std::cmatch' is a}}
  std::cmp_equal; // expected-error {{no member}} expected-note {{'std::cmp_equal' is defined in}} expected-note {{'std::cmp_equal' is a}}
  std::cmp_greater; // expected-error {{no member}} expected-note {{'std::cmp_greater' is defined in}} expected-note {{'std::cmp_greater' is a}}
  std::cmp_greater_equal; // expected-error {{no member}} expected-note {{'std::cmp_greater_equal' is defined in}} expected-note {{'std::cmp_greater_equal' is a}}
  std::cmp_less; // expected-error {{no member}} expected-note {{'std::cmp_less' is defined in}} expected-note {{'std::cmp_less' is a}}
  std::cmp_less_equal; // expected-error {{no member}} expected-note {{'std::cmp_less_equal' is defined in}} expected-note {{'std::cmp_less_equal' is a}}
  std::cmp_not_equal; // expected-error {{no member}} expected-note {{'std::cmp_not_equal' is defined in}} expected-note {{'std::cmp_not_equal' is a}}
  std::codecvt; // expected-error {{no member}} expected-note {{'std::codecvt' is defined in}}
  std::codecvt_base; // expected-error {{no member}} expected-note {{'std::codecvt_base' is defined in}}
  std::codecvt_byname; // expected-error {{no member}} expected-note {{'std::codecvt_byname' is defined in}}
  std::codecvt_mode; // expected-error {{no member}} expected-note {{'std::codecvt_mode' is defined in}} expected-note {{'std::codecvt_mode' is a}}
  std::codecvt_utf16; // expected-error {{no member}} expected-note {{'std::codecvt_utf16' is defined in}} expected-note {{'std::codecvt_utf16' is a}}
  std::codecvt_utf8; // expected-error {{no member}} expected-note {{'std::codecvt_utf8' is defined in}} expected-note {{'std::codecvt_utf8' is a}}
  std::codecvt_utf8_utf16; // expected-error {{no member}} expected-note {{'std::codecvt_utf8_utf16' is defined in}} expected-note {{'std::codecvt_utf8_utf16' is a}}
  std::collate; // expected-error {{no member}} expected-note {{'std::collate' is defined in}}
  std::collate_byname; // expected-error {{no member}} expected-note {{'std::collate_byname' is defined in}}
  std::common_comparison_category; // expected-error {{no member}} expected-note {{'std::common_comparison_category' is defined in}} expected-note {{'std::common_comparison_category' is a}}
  std::common_comparison_category_t; // expected-error {{no member}} expected-note {{'std::common_comparison_category_t' is defined in}} expected-note {{'std::common_comparison_category_t' is a}}
  std::common_iterator; // expected-error {{no member}} expected-note {{'std::common_iterator' is defined in}} expected-note {{'std::common_iterator' is a}}
  std::common_reference; // expected-error {{no member}} expected-note {{'std::common_reference' is defined in}} expected-note {{'std::common_reference' is a}}
  std::common_reference_t; // expected-error {{no member}} expected-note {{'std::common_reference_t' is defined in}} expected-note {{'std::common_reference_t' is a}}
  std::common_reference_with; // expected-error {{no member}} expected-note {{'std::common_reference_with' is defined in}} expected-note {{'std::common_reference_with' is a}}
  std::common_type; // expected-error {{no member}} expected-note {{'std::common_type' is defined in}} expected-note {{'std::common_type' is a}}
  std::common_type_t; // expected-error {{no member}} expected-note {{'std::common_type_t' is defined in}} expected-note {{'std::common_type_t' is a}}
  std::common_with; // expected-error {{no member}} expected-note {{'std::common_with' is defined in}} expected-note {{'std::common_with' is a}}
  std::comp_ellint_1; // expected-error {{no member}} expected-note {{'std::comp_ellint_1' is defined in}} expected-note {{'std::comp_ellint_1' is a}}
  std::comp_ellint_1f; // expected-error {{no member}} expected-note {{'std::comp_ellint_1f' is defined in}} expected-note {{'std::comp_ellint_1f' is a}}
  std::comp_ellint_1l; // expected-error {{no member}} expected-note {{'std::comp_ellint_1l' is defined in}} expected-note {{'std::comp_ellint_1l' is a}}
  std::comp_ellint_2; // expected-error {{no member}} expected-note {{'std::comp_ellint_2' is defined in}} expected-note {{'std::comp_ellint_2' is a}}
  std::comp_ellint_2f; // expected-error {{no member}} expected-note {{'std::comp_ellint_2f' is defined in}} expected-note {{'std::comp_ellint_2f' is a}}
  std::comp_ellint_2l; // expected-error {{no member}} expected-note {{'std::comp_ellint_2l' is defined in}} expected-note {{'std::comp_ellint_2l' is a}}
  std::comp_ellint_3; // expected-error {{no member}} expected-note {{'std::comp_ellint_3' is defined in}} expected-note {{'std::comp_ellint_3' is a}}
  std::comp_ellint_3f; // expected-error {{no member}} expected-note {{'std::comp_ellint_3f' is defined in}} expected-note {{'std::comp_ellint_3f' is a}}
  std::comp_ellint_3l; // expected-error {{no member}} expected-note {{'std::comp_ellint_3l' is defined in}} expected-note {{'std::comp_ellint_3l' is a}}
  std::compare_partial_order_fallback; // expected-error {{no member}} expected-note {{'std::compare_partial_order_fallback' is defined in}} expected-note {{'std::compare_partial_order_fallback' is a}}
  std::compare_strong_order_fallback; // expected-error {{no member}} expected-note {{'std::compare_strong_order_fallback' is defined in}} expected-note {{'std::compare_strong_order_fallback' is a}}
  std::compare_three_way_result; // expected-error {{no member}} expected-note {{'std::compare_three_way_result' is defined in}} expected-note {{'std::compare_three_way_result' is a}}
  std::compare_three_way_result_t; // expected-error {{no member}} expected-note {{'std::compare_three_way_result_t' is defined in}} expected-note {{'std::compare_three_way_result_t' is a}}
  std::compare_weak_order_fallback; // expected-error {{no member}} expected-note {{'std::compare_weak_order_fallback' is defined in}} expected-note {{'std::compare_weak_order_fallback' is a}}
  std::complex; // expected-error {{no member}} expected-note {{'std::complex' is defined in}}
  std::condition_variable; // expected-error {{no member}} expected-note {{'std::condition_variable' is defined in}} expected-note {{'std::condition_variable' is a}}
  std::condition_variable_any; // expected-error {{no member}} expected-note {{'std::condition_variable_any' is defined in}} expected-note {{'std::condition_variable_any' is a}}
  std::conditional; // expected-error {{no member}} expected-note {{'std::conditional' is defined in}} expected-note {{'std::conditional' is a}}
  std::conditional_t; // expected-error {{no member}} expected-note {{'std::conditional_t' is defined in}} expected-note {{'std::conditional_t' is a}}
  std::conj; // expected-error {{no member}} expected-note {{'std::conj' is defined in}}
  std::conjunction; // expected-error {{no member}} expected-note {{'std::conjunction' is defined in}} expected-note {{'std::conjunction' is a}}
  std::conjunction_v; // expected-error {{no member}} expected-note {{'std::conjunction_v' is defined in}} expected-note {{'std::conjunction_v' is a}}
  std::const_iterator; // expected-error {{no member}} expected-note {{'std::const_iterator' is defined in}} expected-note {{'std::const_iterator' is a}}
  std::const_mem_fun1_ref_t; // expected-error {{no member}} expected-note {{'std::const_mem_fun1_ref_t' is defined in}}
  std::const_mem_fun1_t; // expected-error {{no member}} expected-note {{'std::const_mem_fun1_t' is defined in}}
  std::const_mem_fun_ref_t; // expected-error {{no member}} expected-note {{'std::const_mem_fun_ref_t' is defined in}}
  std::const_mem_fun_t; // expected-error {{no member}} expected-note {{'std::const_mem_fun_t' is defined in}}
  std::const_pointer_cast; // expected-error {{no member}} expected-note {{'std::const_pointer_cast' is defined in}} expected-note {{'std::const_pointer_cast' is a}}
  std::const_sentinel; // expected-error {{no member}} expected-note {{'std::const_sentinel' is defined in}} expected-note {{'std::const_sentinel' is a}}
  std::construct_at; // expected-error {{no member}} expected-note {{'std::construct_at' is defined in}} expected-note {{'std::construct_at' is a}}
  std::constructible_from; // expected-error {{no member}} expected-note {{'std::constructible_from' is defined in}} expected-note {{'std::constructible_from' is a}}
  std::contiguous_iterator; // expected-error {{no member}} expected-note {{'std::contiguous_iterator' is defined in}} expected-note {{'std::contiguous_iterator' is a}}
  std::contiguous_iterator_tag; // expected-error {{no member}} expected-note {{'std::contiguous_iterator_tag' is defined in}} expected-note {{'std::contiguous_iterator_tag' is a}}
  std::convertible_to; // expected-error {{no member}} expected-note {{'std::convertible_to' is defined in}} expected-note {{'std::convertible_to' is a}}
  std::copy; // expected-error {{no member}} expected-note {{'std::copy' is defined in}}
  std::copy_backward; // expected-error {{no member}} expected-note {{'std::copy_backward' is defined in}}
  std::copy_constructible; // expected-error {{no member}} expected-note {{'std::copy_constructible' is defined in}} expected-note {{'std::copy_constructible' is a}}
  std::copy_if; // expected-error {{no member}} expected-note {{'std::copy_if' is defined in}} expected-note {{'std::copy_if' is a}}
  std::copy_n; // expected-error {{no member}} expected-note {{'std::copy_n' is defined in}} expected-note {{'std::copy_n' is a}}
  std::copyable; // expected-error {{no member}} expected-note {{'std::copyable' is defined in}} expected-note {{'std::copyable' is a}}
  std::copyable_function; // expected-error {{no member}} expected-note {{'std::copyable_function' is defined in}} expected-note {{'std::copyable_function' is a}}
  std::copysignf; // expected-error {{no member}} expected-note {{'std::copysignf' is defined in}} expected-note {{'std::copysignf' is a}}
  std::copysignl; // expected-error {{no member}} expected-note {{'std::copysignl' is defined in}} expected-note {{'std::copysignl' is a}}
  std::coroutine_handle; // expected-error {{no member}} expected-note {{'std::coroutine_handle' is defined in}} expected-note {{'std::coroutine_handle' is a}}
  std::coroutine_traits; // expected-error {{no member}} expected-note {{'std::coroutine_traits' is defined in}} expected-note {{'std::coroutine_traits' is a}}
  std::cosf; // expected-error {{no member}} expected-note {{'std::cosf' is defined in}} expected-note {{'std::cosf' is a}}
  std::coshf; // expected-error {{no member}} expected-note {{'std::coshf' is defined in}} expected-note {{'std::coshf' is a}}
  std::coshl; // expected-error {{no member}} expected-note {{'std::coshl' is defined in}} expected-note {{'std::coshl' is a}}
  std::cosl; // expected-error {{no member}} expected-note {{'std::cosl' is defined in}} expected-note {{'std::cosl' is a}}
  std::count; // expected-error {{no member}} expected-note {{'std::count' is defined in}}
  std::count_if; // expected-error {{no member}} expected-note {{'std::count_if' is defined in}}
  std::counted_iterator; // expected-error {{no member}} expected-note {{'std::counted_iterator' is defined in}} expected-note {{'std::counted_iterator' is a}}
  std::counting_semaphore; // expected-error {{no member}} expected-note {{'std::counting_semaphore' is defined in}} expected-note {{'std::counting_semaphore' is a}}
  std::countl_one; // expected-error {{no member}} expected-note {{'std::countl_one' is defined in}} expected-note {{'std::countl_one' is a}}
  std::countl_zero; // expected-error {{no member}} expected-note {{'std::countl_zero' is defined in}} expected-note {{'std::countl_zero' is a}}
  std::countr_one; // expected-error {{no member}} expected-note {{'std::countr_one' is defined in}} expected-note {{'std::countr_one' is a}}
  std::countr_zero; // expected-error {{no member}} expected-note {{'std::countr_zero' is defined in}} expected-note {{'std::countr_zero' is a}}
  std::cout; // expected-error {{no member}} expected-note {{'std::cout' is defined in}}
  std::cref; // expected-error {{no member}} expected-note {{'std::cref' is defined in}} expected-note {{'std::cref' is a}}
  std::cregex_iterator; // expected-error {{no member}} expected-note {{'std::cregex_iterator' is defined in}} expected-note {{'std::cregex_iterator' is a}}
  std::cregex_token_iterator; // expected-error {{no member}} expected-note {{'std::cregex_token_iterator' is defined in}} expected-note {{'std::cregex_token_iterator' is a}}
  std::csub_match; // expected-error {{no member}} expected-note {{'std::csub_match' is defined in}} expected-note {{'std::csub_match' is a}}
  std::ctime; // expected-error {{no member}} expected-note {{'std::ctime' is defined in}}
  std::ctype; // expected-error {{no member}} expected-note {{'std::ctype' is defined in}}
  std::ctype_base; // expected-error {{no member}} expected-note {{'std::ctype_base' is defined in}}
  std::ctype_byname; // expected-error {{no member}} expected-note {{'std::ctype_byname' is defined in}}
  std::current_exception; // expected-error {{no member}} expected-note {{'std::current_exception' is defined in}} expected-note {{'std::current_exception' is a}}
  std::cv_status; // expected-error {{no member}} expected-note {{'std::cv_status' is defined in}} expected-note {{'std::cv_status' is a}}
  std::cyl_bessel_i; // expected-error {{no member}} expected-note {{'std::cyl_bessel_i' is defined in}} expected-note {{'std::cyl_bessel_i' is a}}
  std::cyl_bessel_if; // expected-error {{no member}} expected-note {{'std::cyl_bessel_if' is defined in}} expected-note {{'std::cyl_bessel_if' is a}}
  std::cyl_bessel_il; // expected-error {{no member}} expected-note {{'std::cyl_bessel_il' is defined in}} expected-note {{'std::cyl_bessel_il' is a}}
  std::cyl_bessel_j; // expected-error {{no member}} expected-note {{'std::cyl_bessel_j' is defined in}} expected-note {{'std::cyl_bessel_j' is a}}
  std::cyl_bessel_jf; // expected-error {{no member}} expected-note {{'std::cyl_bessel_jf' is defined in}} expected-note {{'std::cyl_bessel_jf' is a}}
  std::cyl_bessel_jl; // expected-error {{no member}} expected-note {{'std::cyl_bessel_jl' is defined in}} expected-note {{'std::cyl_bessel_jl' is a}}
  std::cyl_bessel_k; // expected-error {{no member}} expected-note {{'std::cyl_bessel_k' is defined in}} expected-note {{'std::cyl_bessel_k' is a}}
  std::cyl_bessel_kf; // expected-error {{no member}} expected-note {{'std::cyl_bessel_kf' is defined in}} expected-note {{'std::cyl_bessel_kf' is a}}
  std::cyl_bessel_kl; // expected-error {{no member}} expected-note {{'std::cyl_bessel_kl' is defined in}} expected-note {{'std::cyl_bessel_kl' is a}}
  std::cyl_neumann; // expected-error {{no member}} expected-note {{'std::cyl_neumann' is defined in}} expected-note {{'std::cyl_neumann' is a}}
  std::cyl_neumannf; // expected-error {{no member}} expected-note {{'std::cyl_neumannf' is defined in}} expected-note {{'std::cyl_neumannf' is a}}
  std::cyl_neumannl; // expected-error {{no member}} expected-note {{'std::cyl_neumannl' is defined in}} expected-note {{'std::cyl_neumannl' is a}}
  std::dec; // expected-error {{no member}} expected-note {{'std::dec' is defined in}}
  std::dec; // expected-error {{no member}} expected-note {{'std::dec' is defined in}}
  std::deca; // expected-error {{no member}} expected-note {{'std::deca' is defined in}} expected-note {{'std::deca' is a}}
  std::decay; // expected-error {{no member}} expected-note {{'std::decay' is defined in}} expected-note {{'std::decay' is a}}
  std::decay_t; // expected-error {{no member}} expected-note {{'std::decay_t' is defined in}} expected-note {{'std::decay_t' is a}}
  std::deci; // expected-error {{no member}} expected-note {{'std::deci' is defined in}} expected-note {{'std::deci' is a}}
  std::declare_no_pointers; // expected-error {{no member}} expected-note {{'std::declare_no_pointers' is defined in}} expected-note {{'std::declare_no_pointers' is a}}
  std::declare_reachable; // expected-error {{no member}} expected-note {{'std::declare_reachable' is defined in}} expected-note {{'std::declare_reachable' is a}}
  std::declval; // expected-error {{no member}} expected-note {{'std::declval' is defined in}} expected-note {{'std::declval' is a}}
  std::default_accessor; // expected-error {{no member}} expected-note {{'std::default_accessor' is defined in}} expected-note {{'std::default_accessor' is a}}
  std::default_delete; // expected-error {{no member}} expected-note {{'std::default_delete' is defined in}} expected-note {{'std::default_delete' is a}}
  std::default_initializable; // expected-error {{no member}} expected-note {{'std::default_initializable' is defined in}} expected-note {{'std::default_initializable' is a}}
  std::default_random_engine; // expected-error {{no member}} expected-note {{'std::default_random_engine' is defined in}} expected-note {{'std::default_random_engine' is a}}
  std::default_searcher; // expected-error {{no member}} expected-note {{'std::default_searcher' is defined in}} expected-note {{'std::default_searcher' is a}}
  std::default_sentinel; // expected-error {{no member}} expected-note {{'std::default_sentinel' is defined in}} expected-note {{'std::default_sentinel' is a}}
  std::default_sentinel_t; // expected-error {{no member}} expected-note {{'std::default_sentinel_t' is defined in}} expected-note {{'std::default_sentinel_t' is a}}
  std::defaultfloat; // expected-error {{no member}} expected-note {{'std::defaultfloat' is defined in}} expected-note {{'std::defaultfloat' is a}}
  std::defaultfloat; // expected-error {{no member}} expected-note {{'std::defaultfloat' is defined in}} expected-note {{'std::defaultfloat' is a}}
  std::defer_lock; // expected-error {{no member}} expected-note {{'std::defer_lock' is defined in}} expected-note {{'std::defer_lock' is a}}
  std::defer_lock_t; // expected-error {{no member}} expected-note {{'std::defer_lock_t' is defined in}} expected-note {{'std::defer_lock_t' is a}}
  std::denorm_absent; // expected-error {{no member}} expected-note {{'std::denorm_absent' is defined in}}
  std::denorm_indeterminate; // expected-error {{no member}} expected-note {{'std::denorm_indeterminate' is defined in}}
  std::denorm_present; // expected-error {{no member}} expected-note {{'std::denorm_present' is defined in}}
  std::deque; // expected-error {{no member}} expected-note {{'std::deque' is defined in}}
  std::derived_from; // expected-error {{no member}} expected-note {{'std::derived_from' is defined in}} expected-note {{'std::derived_from' is a}}
  std::destroy; // expected-error {{no member}} expected-note {{'std::destroy' is defined in}} expected-note {{'std::destroy' is a}}
  std::destroy_at; // expected-error {{no member}} expected-note {{'std::destroy_at' is defined in}} expected-note {{'std::destroy_at' is a}}
  std::destroy_n; // expected-error {{no member}} expected-note {{'std::destroy_n' is defined in}} expected-note {{'std::destroy_n' is a}}
  std::destroying_delete; // expected-error {{no member}} expected-note {{'std::destroying_delete' is defined in}} expected-note {{'std::destroying_delete' is a}}
  std::destroying_delete_t; // expected-error {{no member}} expected-note {{'std::destroying_delete_t' is defined in}} expected-note {{'std::destroying_delete_t' is a}}
  std::destructible; // expected-error {{no member}} expected-note {{'std::destructible' is defined in}} expected-note {{'std::destructible' is a}}
  std::dextents; // expected-error {{no member}} expected-note {{'std::dextents' is defined in}} expected-note {{'std::dextents' is a}}
  std::difftime; // expected-error {{no member}} expected-note {{'std::difftime' is defined in}}
  std::dims; // expected-error {{no member}} expected-note {{'std::dims' is defined in}} expected-note {{'std::dims' is a}}
  std::disable_sized_sentinel_for; // expected-error {{no member}} expected-note {{'std::disable_sized_sentinel_for' is defined in}} expected-note {{'std::disable_sized_sentinel_for' is a}}
  std::discard_block_engine; // expected-error {{no member}} expected-note {{'std::discard_block_engine' is defined in}} expected-note {{'std::discard_block_engine' is a}}
  std::discrete_distribution; // expected-error {{no member}} expected-note {{'std::discrete_distribution' is defined in}} expected-note {{'std::discrete_distribution' is a}}
  std::disjunction; // expected-error {{no member}} expected-note {{'std::disjunction' is defined in}} expected-note {{'std::disjunction' is a}}
  std::disjunction_v; // expected-error {{no member}} expected-note {{'std::disjunction_v' is defined in}} expected-note {{'std::disjunction_v' is a}}
  std::distance; // expected-error {{no member}} expected-note {{'std::distance' is defined in}}
  std::div_sat; // expected-error {{no member}} expected-note {{'std::div_sat' is defined in}} expected-note {{'std::div_sat' is a}}
  std::div_t; // expected-error {{no member}} expected-note {{'std::div_t' is defined in}}
  std::divides; // expected-error {{no member}} expected-note {{'std::divides' is defined in}}
  std::domain_error; // expected-error {{no member}} expected-note {{'std::domain_error' is defined in}}
  std::double_t; // expected-error {{no member}} expected-note {{'std::double_t' is defined in}} expected-note {{'std::double_t' is a}}
  std::dynamic_extent; // expected-error {{no member}} expected-note {{'std::dynamic_extent' is defined in}} expected-note {{'std::dynamic_extent' is a}}
  std::dynamic_pointer_cast; // expected-error {{no member}} expected-note {{'std::dynamic_pointer_cast' is defined in}} expected-note {{'std::dynamic_pointer_cast' is a}}
  std::ellint_1; // expected-error {{no member}} expected-note {{'std::ellint_1' is defined in}} expected-note {{'std::ellint_1' is a}}
  std::ellint_1f; // expected-error {{no member}} expected-note {{'std::ellint_1f' is defined in}} expected-note {{'std::ellint_1f' is a}}
  std::ellint_1l; // expected-error {{no member}} expected-note {{'std::ellint_1l' is defined in}} expected-note {{'std::ellint_1l' is a}}
  std::ellint_2; // expected-error {{no member}} expected-note {{'std::ellint_2' is defined in}} expected-note {{'std::ellint_2' is a}}
  std::ellint_2f; // expected-error {{no member}} expected-note {{'std::ellint_2f' is defined in}} expected-note {{'std::ellint_2f' is a}}
  std::ellint_2l; // expected-error {{no member}} expected-note {{'std::ellint_2l' is defined in}} expected-note {{'std::ellint_2l' is a}}
  std::ellint_3; // expected-error {{no member}} expected-note {{'std::ellint_3' is defined in}} expected-note {{'std::ellint_3' is a}}
  std::ellint_3f; // expected-error {{no member}} expected-note {{'std::ellint_3f' is defined in}} expected-note {{'std::ellint_3f' is a}}
  std::ellint_3l; // expected-error {{no member}} expected-note {{'std::ellint_3l' is defined in}} expected-note {{'std::ellint_3l' is a}}
  std::emit_on_flush; // expected-error {{no member}} expected-note {{'std::emit_on_flush' is defined in}} expected-note {{'std::emit_on_flush' is a}}
  std::emit_on_flush; // expected-error {{no member}} expected-note {{'std::emit_on_flush' is defined in}} expected-note {{'std::emit_on_flush' is a}}
  std::enable_if; // expected-error {{no member}} expected-note {{'std::enable_if' is defined in}} expected-note {{'std::enable_if' is a}}
  std::enable_if_t; // expected-error {{no member}} expected-note {{'std::enable_if_t' is defined in}} expected-note {{'std::enable_if_t' is a}}
  std::enable_nonlocking_formatter_optimization; // expected-error {{no member}} expected-note {{'std::enable_nonlocking_formatter_optimization' is defined in}} expected-note {{'std::enable_nonlocking_formatter_optimization' is a}}
  std::enable_shared_from_this; // expected-error {{no member}} expected-note {{'std::enable_shared_from_this' is defined in}} expected-note {{'std::enable_shared_from_this' is a}}
  std::endian; // expected-error {{no member}} expected-note {{'std::endian' is defined in}} expected-note {{'std::endian' is a}}
  std::endl; // expected-error {{no member}} expected-note {{'std::endl' is defined in}}
  std::endl; // expected-error {{no member}} expected-note {{'std::endl' is defined in}}
  std::ends; // expected-error {{no member}} expected-note {{'std::ends' is defined in}}
  std::ends; // expected-error {{no member}} expected-note {{'std::ends' is defined in}}
  std::equal; // expected-error {{no member}} expected-note {{'std::equal' is defined in}}
  std::equal_range; // expected-error {{no member}} expected-note {{'std::equal_range' is defined in}}
  std::equal_to; // expected-error {{no member}} expected-note {{'std::equal_to' is defined in}}
  std::equality_comparable; // expected-error {{no member}} expected-note {{'std::equality_comparable' is defined in}} expected-note {{'std::equality_comparable' is a}}
  std::equality_comparable_with; // expected-error {{no member}} expected-note {{'std::equality_comparable_with' is defined in}} expected-note {{'std::equality_comparable_with' is a}}
  std::equivalence_relation; // expected-error {{no member}} expected-note {{'std::equivalence_relation' is defined in}} expected-note {{'std::equivalence_relation' is a}}
  std::erfcf; // expected-error {{no member}} expected-note {{'std::erfcf' is defined in}} expected-note {{'std::erfcf' is a}}
  std::erfcl; // expected-error {{no member}} expected-note {{'std::erfcl' is defined in}} expected-note {{'std::erfcl' is a}}
  std::erff; // expected-error {{no member}} expected-note {{'std::erff' is defined in}} expected-note {{'std::erff' is a}}
  std::erfl; // expected-error {{no member}} expected-note {{'std::erfl' is defined in}} expected-note {{'std::erfl' is a}}
  std::errc; // expected-error {{no member}} expected-note {{'std::errc' is defined in}} expected-note {{'std::errc' is a}}
  std::error_category; // expected-error {{no member}} expected-note {{'std::error_category' is defined in}} expected-note {{'std::error_category' is a}}
  std::error_code; // expected-error {{no member}} expected-note {{'std::error_code' is defined in}} expected-note {{'std::error_code' is a}}
  std::error_condition; // expected-error {{no member}} expected-note {{'std::error_condition' is defined in}} expected-note {{'std::error_condition' is a}}
  std::exa; // expected-error {{no member}} expected-note {{'std::exa' is defined in}} expected-note {{'std::exa' is a}}
  std::exception; // expected-error {{no member}} expected-note {{'std::exception' is defined in}}
  std::exception_ptr; // expected-error {{no member}} expected-note {{'std::exception_ptr' is defined in}} expected-note {{'std::exception_ptr' is a}}
  std::exchange; // expected-error {{no member}} expected-note {{'std::exchange' is defined in}} expected-note {{'std::exchange' is a}}
  std::exclusive_scan; // expected-error {{no member}} expected-note {{'std::exclusive_scan' is defined in}} expected-note {{'std::exclusive_scan' is a}}
  std::exit; // expected-error {{no member}} expected-note {{'std::exit' is defined in}}
  std::exp2f; // expected-error {{no member}} expected-note {{'std::exp2f' is defined in}} expected-note {{'std::exp2f' is a}}
  std::exp2l; // expected-error {{no member}} expected-note {{'std::exp2l' is defined in}} expected-note {{'std::exp2l' is a}}
  std::expected; // expected-error {{no member}} expected-note {{'std::expected' is defined in}} expected-note {{'std::expected' is a}}
  std::expf; // expected-error {{no member}} expected-note {{'std::expf' is defined in}} expected-note {{'std::expf' is a}}
  std::expint; // expected-error {{no member}} expected-note {{'std::expint' is defined in}} expected-note {{'std::expint' is a}}
  std::expintf; // expected-error {{no member}} expected-note {{'std::expintf' is defined in}} expected-note {{'std::expintf' is a}}
  std::expintl; // expected-error {{no member}} expected-note {{'std::expintl' is defined in}} expected-note {{'std::expintl' is a}}
  std::expl; // expected-error {{no member}} expected-note {{'std::expl' is defined in}} expected-note {{'std::expl' is a}}
  std::expm1f; // expected-error {{no member}} expected-note {{'std::expm1f' is defined in}} expected-note {{'std::expm1f' is a}}
  std::expm1l; // expected-error {{no member}} expected-note {{'std::expm1l' is defined in}} expected-note {{'std::expm1l' is a}}
  std::exponential_distribution; // expected-error {{no member}} expected-note {{'std::exponential_distribution' is defined in}} expected-note {{'std::exponential_distribution' is a}}
  std::extent; // expected-error {{no member}} expected-note {{'std::extent' is defined in}} expected-note {{'std::extent' is a}}
  std::extent_v; // expected-error {{no member}} expected-note {{'std::extent_v' is defined in}} expected-note {{'std::extent_v' is a}}
  std::extents; // expected-error {{no member}} expected-note {{'std::extents' is defined in}} expected-note {{'std::extents' is a}}
  std::extreme_value_distribution; // expected-error {{no member}} expected-note {{'std::extreme_value_distribution' is defined in}} expected-note {{'std::extreme_value_distribution' is a}}
  std::fabs; // expected-error {{no member}} expected-note {{'std::fabs' is defined in}}
  std::fabsf; // expected-error {{no member}} expected-note {{'std::fabsf' is defined in}} expected-note {{'std::fabsf' is a}}
  std::fabsl; // expected-error {{no member}} expected-note {{'std::fabsl' is defined in}} expected-note {{'std::fabsl' is a}}
  std::false_type; // expected-error {{no member}} expected-note {{'std::false_type' is defined in}} expected-note {{'std::false_type' is a}}
  std::fclose; // expected-error {{no member}} expected-note {{'std::fclose' is defined in}}
  std::fdimf; // expected-error {{no member}} expected-note {{'std::fdimf' is defined in}} expected-note {{'std::fdimf' is a}}
  std::fdiml; // expected-error {{no member}} expected-note {{'std::fdiml' is defined in}} expected-note {{'std::fdiml' is a}}
  std::feclearexcept; // expected-error {{no member}} expected-note {{'std::feclearexcept' is defined in}} expected-note {{'std::feclearexcept' is a}}
  std::fegetenv; // expected-error {{no member}} expected-note {{'std::fegetenv' is defined in}} expected-note {{'std::fegetenv' is a}}
  std::fegetexceptflag; // expected-error {{no member}} expected-note {{'std::fegetexceptflag' is defined in}} expected-note {{'std::fegetexceptflag' is a}}
  std::fegetround; // expected-error {{no member}} expected-note {{'std::fegetround' is defined in}} expected-note {{'std::fegetround' is a}}
  std::feholdexcept; // expected-error {{no member}} expected-note {{'std::feholdexcept' is defined in}} expected-note {{'std::feholdexcept' is a}}
  std::femto; // expected-error {{no member}} expected-note {{'std::femto' is defined in}} expected-note {{'std::femto' is a}}
  std::fenv_t; // expected-error {{no member}} expected-note {{'std::fenv_t' is defined in}} expected-note {{'std::fenv_t' is a}}
  std::feof; // expected-error {{no member}} expected-note {{'std::feof' is defined in}}
  std::feraiseexcept; // expected-error {{no member}} expected-note {{'std::feraiseexcept' is defined in}} expected-note {{'std::feraiseexcept' is a}}
  std::ferror; // expected-error {{no member}} expected-note {{'std::ferror' is defined in}}
  std::fesetenv; // expected-error {{no member}} expected-note {{'std::fesetenv' is defined in}} expected-note {{'std::fesetenv' is a}}
  std::fesetexceptflag; // expected-error {{no member}} expected-note {{'std::fesetexceptflag' is defined in}} expected-note {{'std::fesetexceptflag' is a}}
  std::fesetround; // expected-error {{no member}} expected-note {{'std::fesetround' is defined in}} expected-note {{'std::fesetround' is a}}
  std::fetestexcept; // expected-error {{no member}} expected-note {{'std::fetestexcept' is defined in}} expected-note {{'std::fetestexcept' is a}}
  std::feupdateenv; // expected-error {{no member}} expected-note {{'std::feupdateenv' is defined in}} expected-note {{'std::feupdateenv' is a}}
  std::fexcept_t; // expected-error {{no member}} expected-note {{'std::fexcept_t' is defined in}} expected-note {{'std::fexcept_t' is a}}
  std::fflush; // expected-error {{no member}} expected-note {{'std::fflush' is defined in}}
  std::fgetc; // expected-error {{no member}} expected-note {{'std::fgetc' is defined in}}
  std::fgetpos; // expected-error {{no member}} expected-note {{'std::fgetpos' is defined in}}
  std::fgets; // expected-error {{no member}} expected-note {{'std::fgets' is defined in}}
  std::fgetwc; // expected-error {{no member}} expected-note {{'std::fgetwc' is defined in}}
  std::fgetws; // expected-error {{no member}} expected-note {{'std::fgetws' is defined in}}
  std::filebuf; // expected-error {{no member}} expected-note {{'std::filebuf' is defined in}}
  std::filebuf; // expected-error {{no member}} expected-note {{'std::filebuf' is defined in}}
  std::fill; // expected-error {{no member}} expected-note {{'std::fill' is defined in}}
  std::fill_n; // expected-error {{no member}} expected-note {{'std::fill_n' is defined in}}
  std::find; // expected-error {{no member}} expected-note {{'std::find' is defined in}}
  std::find_end; // expected-error {{no member}} expected-note {{'std::find_end' is defined in}}
  std::find_first_of; // expected-error {{no member}} expected-note {{'std::find_first_of' is defined in}}
  std::find_if; // expected-error {{no member}} expected-note {{'std::find_if' is defined in}}
  std::find_if_not; // expected-error {{no member}} expected-note {{'std::find_if_not' is defined in}} expected-note {{'std::find_if_not' is a}}
  std::fisher_f_distribution; // expected-error {{no member}} expected-note {{'std::fisher_f_distribution' is defined in}} expected-note {{'std::fisher_f_distribution' is a}}
  std::fixed; // expected-error {{no member}} expected-note {{'std::fixed' is defined in}}
  std::fixed; // expected-error {{no member}} expected-note {{'std::fixed' is defined in}}
  std::flat_map; // expected-error {{no member}} expected-note {{'std::flat_map' is defined in}} expected-note {{'std::flat_map' is a}}
  std::flat_multimap; // expected-error {{no member}} expected-note {{'std::flat_multimap' is defined in}} expected-note {{'std::flat_multimap' is a}}
  std::flat_multiset; // expected-error {{no member}} expected-note {{'std::flat_multiset' is defined in}} expected-note {{'std::flat_multiset' is a}}
  std::flat_set; // expected-error {{no member}} expected-note {{'std::flat_set' is defined in}} expected-note {{'std::flat_set' is a}}
  std::float_denorm_style; // expected-error {{no member}} expected-note {{'std::float_denorm_style' is defined in}}
  std::float_round_style; // expected-error {{no member}} expected-note {{'std::float_round_style' is defined in}}
  std::float_t; // expected-error {{no member}} expected-note {{'std::float_t' is defined in}} expected-note {{'std::float_t' is a}}
  std::floating_point; // expected-error {{no member}} expected-note {{'std::floating_point' is defined in}} expected-note {{'std::floating_point' is a}}
  std::floorf; // expected-error {{no member}} expected-note {{'std::floorf' is defined in}} expected-note {{'std::floorf' is a}}
  std::floorl; // expected-error {{no member}} expected-note {{'std::floorl' is defined in}} expected-note {{'std::floorl' is a}}
  std::flush; // expected-error {{no member}} expected-note {{'std::flush' is defined in}}
  std::flush; // expected-error {{no member}} expected-note {{'std::flush' is defined in}}
  std::flush_emit; // expected-error {{no member}} expected-note {{'std::flush_emit' is defined in}} expected-note {{'std::flush_emit' is a}}
  std::flush_emit; // expected-error {{no member}} expected-note {{'std::flush_emit' is defined in}} expected-note {{'std::flush_emit' is a}}
  std::fma; // expected-error {{no member}} expected-note {{'std::fma' is defined in}} expected-note {{'std::fma' is a}}
  std::fmaf; // expected-error {{no member}} expected-note {{'std::fmaf' is defined in}} expected-note {{'std::fmaf' is a}}
  std::fmal; // expected-error {{no member}} expected-note {{'std::fmal' is defined in}} expected-note {{'std::fmal' is a}}
  std::fmaxf; // expected-error {{no member}} expected-note {{'std::fmaxf' is defined in}} expected-note {{'std::fmaxf' is a}}
  std::fmaxl; // expected-error {{no member}} expected-note {{'std::fmaxl' is defined in}} expected-note {{'std::fmaxl' is a}}
  std::fminf; // expected-error {{no member}} expected-note {{'std::fminf' is defined in}} expected-note {{'std::fminf' is a}}
  std::fminl; // expected-error {{no member}} expected-note {{'std::fminl' is defined in}} expected-note {{'std::fminl' is a}}
  std::fmodf; // expected-error {{no member}} expected-note {{'std::fmodf' is defined in}} expected-note {{'std::fmodf' is a}}
  std::fmodl; // expected-error {{no member}} expected-note {{'std::fmodl' is defined in}} expected-note {{'std::fmodl' is a}}
  std::fopen; // expected-error {{no member}} expected-note {{'std::fopen' is defined in}}
  std::for_each; // expected-error {{no member}} expected-note {{'std::for_each' is defined in}}
  std::for_each_n; // expected-error {{no member}} expected-note {{'std::for_each_n' is defined in}} expected-note {{'std::for_each_n' is a}}
  std::format; // expected-error {{no member}} expected-note {{'std::format' is defined in}} expected-note {{'std::format' is a}}
  std::format_args; // expected-error {{no member}} expected-note {{'std::format_args' is defined in}} expected-note {{'std::format_args' is a}}
  std::format_context; // expected-error {{no member}} expected-note {{'std::format_context' is defined in}} expected-note {{'std::format_context' is a}}
  std::format_error; // expected-error {{no member}} expected-note {{'std::format_error' is defined in}} expected-note {{'std::format_error' is a}}
  std::format_kind; // expected-error {{no member}} expected-note {{'std::format_kind' is defined in}} expected-note {{'std::format_kind' is a}}
  std::format_parse_context; // expected-error {{no member}} expected-note {{'std::format_parse_context' is defined in}} expected-note {{'std::format_parse_context' is a}}
  std::format_string; // expected-error {{no member}} expected-note {{'std::format_string' is defined in}} expected-note {{'std::format_string' is a}}
  std::format_to; // expected-error {{no member}} expected-note {{'std::format_to' is defined in}} expected-note {{'std::format_to' is a}}
  std::format_to_n; // expected-error {{no member}} expected-note {{'std::format_to_n' is defined in}} expected-note {{'std::format_to_n' is a}}
  std::format_to_n_result; // expected-error {{no member}} expected-note {{'std::format_to_n_result' is defined in}} expected-note {{'std::format_to_n_result' is a}}
  std::formattable; // expected-error {{no member}} expected-note {{'std::formattable' is defined in}} expected-note {{'std::formattable' is a}}
  std::formatted_size; // expected-error {{no member}} expected-note {{'std::formatted_size' is defined in}} expected-note {{'std::formatted_size' is a}}
  std::formatter; // expected-error {{no member}} expected-note {{'std::formatter' is defined in}} expected-note {{'std::formatter' is a}}
  std::forward; // expected-error {{no member}} expected-note {{'std::forward' is defined in}} expected-note {{'std::forward' is a}}
  std::forward_as_tuple; // expected-error {{no member}} expected-note {{'std::forward_as_tuple' is defined in}} expected-note {{'std::forward_as_tuple' is a}}
  std::forward_iterator; // expected-error {{no member}} expected-note {{'std::forward_iterator' is defined in}} expected-note {{'std::forward_iterator' is a}}
  std::forward_iterator_tag; // expected-error {{no member}} expected-note {{'std::forward_iterator_tag' is defined in}}
  std::forward_like; // expected-error {{no member}} expected-note {{'std::forward_like' is defined in}} expected-note {{'std::forward_like' is a}}
  std::forward_list; // expected-error {{no member}} expected-note {{'std::forward_list' is defined in}} expected-note {{'std::forward_list' is a}}
  std::fpclassify; // expected-error {{no member}} expected-note {{'std::fpclassify' is defined in}} expected-note {{'std::fpclassify' is a}}
  std::fpos; // expected-error {{no member}} expected-note {{'std::fpos' is defined in}}
  std::fpos; // expected-error {{no member}} expected-note {{'std::fpos' is defined in}}
  std::fpos; // expected-error {{no member}} expected-note {{'std::fpos' is defined in}}
  std::fpos_t; // expected-error {{no member}} expected-note {{'std::fpos_t' is defined in}}
  std::fprintf; // expected-error {{no member}} expected-note {{'std::fprintf' is defined in}}
  std::fputc; // expected-error {{no member}} expected-note {{'std::fputc' is defined in}}
  std::fputs; // expected-error {{no member}} expected-note {{'std::fputs' is defined in}}
  std::fputwc; // expected-error {{no member}} expected-note {{'std::fputwc' is defined in}}
  std::fputws; // expected-error {{no member}} expected-note {{'std::fputws' is defined in}}
  std::fread; // expected-error {{no member}} expected-note {{'std::fread' is defined in}}
  std::free; // expected-error {{no member}} expected-note {{'std::free' is defined in}}
  std::freopen; // expected-error {{no member}} expected-note {{'std::freopen' is defined in}}
  std::frexp; // expected-error {{no member}} expected-note {{'std::frexp' is defined in}}
  std::frexpf; // expected-error {{no member}} expected-note {{'std::frexpf' is defined in}} expected-note {{'std::frexpf' is a}}
  std::frexpl; // expected-error {{no member}} expected-note {{'std::frexpl' is defined in}} expected-note {{'std::frexpl' is a}}
  std::from_chars; // expected-error {{no member}} expected-note {{'std::from_chars' is defined in}} expected-note {{'std::from_chars' is a}}
  std::from_chars_result; // expected-error {{no member}} expected-note {{'std::from_chars_result' is defined in}} expected-note {{'std::from_chars_result' is a}}
  std::from_range; // expected-error {{no member}} expected-note {{'std::from_range' is defined in}} expected-note {{'std::from_range' is a}}
  std::from_range_t; // expected-error {{no member}} expected-note {{'std::from_range_t' is defined in}} expected-note {{'std::from_range_t' is a}}
  std::front_insert_iterator; // expected-error {{no member}} expected-note {{'std::front_insert_iterator' is defined in}}
  std::front_inserter; // expected-error {{no member}} expected-note {{'std::front_inserter' is defined in}}
  std::fscanf; // expected-error {{no member}} expected-note {{'std::fscanf' is defined in}}
  std::fseek; // expected-error {{no member}} expected-note {{'std::fseek' is defined in}}
  std::fsetpos; // expected-error {{no member}} expected-note {{'std::fsetpos' is defined in}}
  std::fstream; // expected-error {{no member}} expected-note {{'std::fstream' is defined in}}
  std::fstream; // expected-error {{no member}} expected-note {{'std::fstream' is defined in}}
  std::ftell; // expected-error {{no member}} expected-note {{'std::ftell' is defined in}}
  std::function; // expected-error {{no member}} expected-note {{'std::function' is defined in}} expected-note {{'std::function' is a}}
  std::function_ref; // expected-error {{no member}} expected-note {{'std::function_ref' is defined in}} expected-note {{'std::function_ref' is a}}
  std::future; // expected-error {{no member}} expected-note {{'std::future' is defined in}} expected-note {{'std::future' is a}}
  std::future_category; // expected-error {{no member}} expected-note {{'std::future_category' is defined in}} expected-note {{'std::future_category' is a}}
  std::future_errc; // expected-error {{no member}} expected-note {{'std::future_errc' is defined in}} expected-note {{'std::future_errc' is a}}
  std::future_error; // expected-error {{no member}} expected-note {{'std::future_error' is defined in}} expected-note {{'std::future_error' is a}}
  std::future_status; // expected-error {{no member}} expected-note {{'std::future_status' is defined in}} expected-note {{'std::future_status' is a}}
  std::fwide; // expected-error {{no member}} expected-note {{'std::fwide' is defined in}}
  std::fwprintf; // expected-error {{no member}} expected-note {{'std::fwprintf' is defined in}}
  std::fwrite; // expected-error {{no member}} expected-note {{'std::fwrite' is defined in}}
  std::fwscanf; // expected-error {{no member}} expected-note {{'std::fwscanf' is defined in}}
  std::gamma_distribution; // expected-error {{no member}} expected-note {{'std::gamma_distribution' is defined in}} expected-note {{'std::gamma_distribution' is a}}
  std::gcd; // expected-error {{no member}} expected-note {{'std::gcd' is defined in}} expected-note {{'std::gcd' is a}}
  std::generate; // expected-error {{no member}} expected-note {{'std::generate' is defined in}}
  std::generate_canonical; // expected-error {{no member}} expected-note {{'std::generate_canonical' is defined in}} expected-note {{'std::generate_canonical' is a}}
  std::generate_n; // expected-error {{no member}} expected-note {{'std::generate_n' is defined in}}
  std::generator; // expected-error {{no member}} expected-note {{'std::generator' is defined in}} expected-note {{'std::generator' is a}}
  std::generic_category; // expected-error {{no member}} expected-note {{'std::generic_category' is defined in}} expected-note {{'std::generic_category' is a}}
  std::geometric_distribution; // expected-error {{no member}} expected-note {{'std::geometric_distribution' is defined in}} expected-note {{'std::geometric_distribution' is a}}
  std::get_deleter; // expected-error {{no member}} expected-note {{'std::get_deleter' is defined in}} expected-note {{'std::get_deleter' is a}}
  std::get_if; // expected-error {{no member}} expected-note {{'std::get_if' is defined in}} expected-note {{'std::get_if' is a}}
  std::get_money; // expected-error {{no member}} expected-note {{'std::get_money' is defined in}} expected-note {{'std::get_money' is a}}
  std::get_new_handler; // expected-error {{no member}} expected-note {{'std::get_new_handler' is defined in}} expected-note {{'std::get_new_handler' is a}}
  std::get_pointer_safety; // expected-error {{no member}} expected-note {{'std::get_pointer_safety' is defined in}} expected-note {{'std::get_pointer_safety' is a}}
  std::get_temporary_buffer; // expected-error {{no member}} expected-note {{'std::get_temporary_buffer' is defined in}}
  std::get_terminate; // expected-error {{no member}} expected-note {{'std::get_terminate' is defined in}} expected-note {{'std::get_terminate' is a}}
  std::get_time; // expected-error {{no member}} expected-note {{'std::get_time' is defined in}} expected-note {{'std::get_time' is a}}
  std::get_unexpected; // expected-error {{no member}} expected-note {{'std::get_unexpected' is defined in}}
  std::getc; // expected-error {{no member}} expected-note {{'std::getc' is defined in}}
  std::getchar; // expected-error {{no member}} expected-note {{'std::getchar' is defined in}}
  std::getenv; // expected-error {{no member}} expected-note {{'std::getenv' is defined in}}
  std::getline; // expected-error {{no member}} expected-note {{'std::getline' is defined in}}
  std::gets; // expected-error {{no member}} expected-note {{'std::gets' is defined in}}
  std::getwc; // expected-error {{no member}} expected-note {{'std::getwc' is defined in}}
  std::getwchar; // expected-error {{no member}} expected-note {{'std::getwchar' is defined in}}
  std::giga; // expected-error {{no member}} expected-note {{'std::giga' is defined in}} expected-note {{'std::giga' is a}}
  std::gmtime; // expected-error {{no member}} expected-note {{'std::gmtime' is defined in}}
  std::greater; // expected-error {{no member}} expected-note {{'std::greater' is defined in}}
  std::greater_equal; // expected-error {{no member}} expected-note {{'std::greater_equal' is defined in}}
  std::gslice; // expected-error {{no member}} expected-note {{'std::gslice' is defined in}}
  std::gslice_array; // expected-error {{no member}} expected-note {{'std::gslice_array' is defined in}}
  std::hardware_constructive_interference_size; // expected-error {{no member}} expected-note {{'std::hardware_constructive_interference_size' is defined in}} expected-note {{'std::hardware_constructive_interference_size' is a}}
  std::hardware_destructive_interference_size; // expected-error {{no member}} expected-note {{'std::hardware_destructive_interference_size' is defined in}} expected-note {{'std::hardware_destructive_interference_size' is a}}
  std::has_facet; // expected-error {{no member}} expected-note {{'std::has_facet' is defined in}}
  std::has_single_bit; // expected-error {{no member}} expected-note {{'std::has_single_bit' is defined in}} expected-note {{'std::has_single_bit' is a}}
  std::has_unique_object_representations; // expected-error {{no member}} expected-note {{'std::has_unique_object_representations' is defined in}} expected-note {{'std::has_unique_object_representations' is a}}
  std::has_unique_object_representations_v; // expected-error {{no member}} expected-note {{'std::has_unique_object_representations_v' is defined in}} expected-note {{'std::has_unique_object_representations_v' is a}}
  std::has_virtual_destructor; // expected-error {{no member}} expected-note {{'std::has_virtual_destructor' is defined in}} expected-note {{'std::has_virtual_destructor' is a}}
  std::has_virtual_destructor_v; // expected-error {{no member}} expected-note {{'std::has_virtual_destructor_v' is defined in}} expected-note {{'std::has_virtual_destructor_v' is a}}
  std::hecto; // expected-error {{no member}} expected-note {{'std::hecto' is defined in}} expected-note {{'std::hecto' is a}}
  std::hermite; // expected-error {{no member}} expected-note {{'std::hermite' is defined in}} expected-note {{'std::hermite' is a}}
  std::hermitef; // expected-error {{no member}} expected-note {{'std::hermitef' is defined in}} expected-note {{'std::hermitef' is a}}
  std::hermitel; // expected-error {{no member}} expected-note {{'std::hermitel' is defined in}} expected-note {{'std::hermitel' is a}}
  std::hex; // expected-error {{no member}} expected-note {{'std::hex' is defined in}}
  std::hex; // expected-error {{no member}} expected-note {{'std::hex' is defined in}}
  std::hexfloat; // expected-error {{no member}} expected-note {{'std::hexfloat' is defined in}} expected-note {{'std::hexfloat' is a}}
  std::hexfloat; // expected-error {{no member}} expected-note {{'std::hexfloat' is defined in}} expected-note {{'std::hexfloat' is a}}
  std::holds_alternative; // expected-error {{no member}} expected-note {{'std::holds_alternative' is defined in}} expected-note {{'std::holds_alternative' is a}}
  std::hypot; // expected-error {{no member}} expected-note {{'std::hypot' is defined in}} expected-note {{'std::hypot' is a}}
  std::hypotf; // expected-error {{no member}} expected-note {{'std::hypotf' is defined in}} expected-note {{'std::hypotf' is a}}
  std::hypotl; // expected-error {{no member}} expected-note {{'std::hypotl' is defined in}} expected-note {{'std::hypotl' is a}}
  std::identity; // expected-error {{no member}} expected-note {{'std::identity' is defined in}} expected-note {{'std::identity' is a}}
  std::ifstream; // expected-error {{no member}} expected-note {{'std::ifstream' is defined in}}
  std::ifstream; // expected-error {{no member}} expected-note {{'std::ifstream' is defined in}}
  std::ilogb; // expected-error {{no member}} expected-note {{'std::ilogb' is defined in}} expected-note {{'std::ilogb' is a}}
  std::ilogbf; // expected-error {{no member}} expected-note {{'std::ilogbf' is defined in}} expected-note {{'std::ilogbf' is a}}
  std::ilogbl; // expected-error {{no member}} expected-note {{'std::ilogbl' is defined in}} expected-note {{'std::ilogbl' is a}}
  std::imag; // expected-error {{no member}} expected-note {{'std::imag' is defined in}}
  std::imaxabs; // expected-error {{no member}} expected-note {{'std::imaxabs' is defined in}} expected-note {{'std::imaxabs' is a}}
  std::imaxdiv; // expected-error {{no member}} expected-note {{'std::imaxdiv' is defined in}} expected-note {{'std::imaxdiv' is a}}
  std::imaxdiv_t; // expected-error {{no member}} expected-note {{'std::imaxdiv_t' is defined in}} expected-note {{'std::imaxdiv_t' is a}}
  std::in_place; // expected-error {{no member}} expected-note {{'std::in_place' is defined in}} expected-note {{'std::in_place' is a}}
  std::in_place_index; // expected-error {{no member}} expected-note {{'std::in_place_index' is defined in}} expected-note {{'std::in_place_index' is a}}
  std::in_place_index_t; // expected-error {{no member}} expected-note {{'std::in_place_index_t' is defined in}} expected-note {{'std::in_place_index_t' is a}}
  std::in_place_t; // expected-error {{no member}} expected-note {{'std::in_place_t' is defined in}} expected-note {{'std::in_place_t' is a}}
  std::in_place_type; // expected-error {{no member}} expected-note {{'std::in_place_type' is defined in}} expected-note {{'std::in_place_type' is a}}
  std::in_place_type_t; // expected-error {{no member}} expected-note {{'std::in_place_type_t' is defined in}} expected-note {{'std::in_place_type_t' is a}}
  std::in_range; // expected-error {{no member}} expected-note {{'std::in_range' is defined in}} expected-note {{'std::in_range' is a}}
  std::includes; // expected-error {{no member}} expected-note {{'std::includes' is defined in}}
  std::inclusive_scan; // expected-error {{no member}} expected-note {{'std::inclusive_scan' is defined in}} expected-note {{'std::inclusive_scan' is a}}
  std::incrementable; // expected-error {{no member}} expected-note {{'std::incrementable' is defined in}} expected-note {{'std::incrementable' is a}}
  std::incrementable_traits; // expected-error {{no member}} expected-note {{'std::incrementable_traits' is defined in}} expected-note {{'std::incrementable_traits' is a}}
  std::independent_bits_engine; // expected-error {{no member}} expected-note {{'std::independent_bits_engine' is defined in}} expected-note {{'std::independent_bits_engine' is a}}
  std::index_sequence; // expected-error {{no member}} expected-note {{'std::index_sequence' is defined in}} expected-note {{'std::index_sequence' is a}}
  std::index_sequence_for; // expected-error {{no member}} expected-note {{'std::index_sequence_for' is defined in}} expected-note {{'std::index_sequence_for' is a}}
  std::indirect_array; // expected-error {{no member}} expected-note {{'std::indirect_array' is defined in}}
  std::indirect_binary_predicate; // expected-error {{no member}} expected-note {{'std::indirect_binary_predicate' is defined in}} expected-note {{'std::indirect_binary_predicate' is a}}
  std::indirect_equivalence_relation; // expected-error {{no member}} expected-note {{'std::indirect_equivalence_relation' is defined in}} expected-note {{'std::indirect_equivalence_relation' is a}}
  std::indirect_result_t; // expected-error {{no member}} expected-note {{'std::indirect_result_t' is defined in}} expected-note {{'std::indirect_result_t' is a}}
  std::indirect_strict_weak_order; // expected-error {{no member}} expected-note {{'std::indirect_strict_weak_order' is defined in}} expected-note {{'std::indirect_strict_weak_order' is a}}
  std::indirect_unary_predicate; // expected-error {{no member}} expected-note {{'std::indirect_unary_predicate' is defined in}} expected-note {{'std::indirect_unary_predicate' is a}}
  std::indirectly_comparable; // expected-error {{no member}} expected-note {{'std::indirectly_comparable' is defined in}} expected-note {{'std::indirectly_comparable' is a}}
  std::indirectly_copyable; // expected-error {{no member}} expected-note {{'std::indirectly_copyable' is defined in}} expected-note {{'std::indirectly_copyable' is a}}
  std::indirectly_copyable_storable; // expected-error {{no member}} expected-note {{'std::indirectly_copyable_storable' is defined in}} expected-note {{'std::indirectly_copyable_storable' is a}}
  std::indirectly_movable; // expected-error {{no member}} expected-note {{'std::indirectly_movable' is defined in}} expected-note {{'std::indirectly_movable' is a}}
  std::indirectly_movable_storable; // expected-error {{no member}} expected-note {{'std::indirectly_movable_storable' is defined in}} expected-note {{'std::indirectly_movable_storable' is a}}
  std::indirectly_readable; // expected-error {{no member}} expected-note {{'std::indirectly_readable' is defined in}} expected-note {{'std::indirectly_readable' is a}}
  std::indirectly_readable_traits; // expected-error {{no member}} expected-note {{'std::indirectly_readable_traits' is defined in}} expected-note {{'std::indirectly_readable_traits' is a}}
  std::indirectly_regular_unary_invocable; // expected-error {{no member}} expected-note {{'std::indirectly_regular_unary_invocable' is defined in}} expected-note {{'std::indirectly_regular_unary_invocable' is a}}
  std::indirectly_swappable; // expected-error {{no member}} expected-note {{'std::indirectly_swappable' is defined in}} expected-note {{'std::indirectly_swappable' is a}}
  std::indirectly_unary_invocable; // expected-error {{no member}} expected-note {{'std::indirectly_unary_invocable' is defined in}} expected-note {{'std::indirectly_unary_invocable' is a}}
  std::indirectly_writable; // expected-error {{no member}} expected-note {{'std::indirectly_writable' is defined in}} expected-note {{'std::indirectly_writable' is a}}
  std::initializer_list; // expected-error {{no member}} expected-note {{'std::initializer_list' is defined in}} expected-note {{'std::initializer_list' is a}}
  std::inner_product; // expected-error {{no member}} expected-note {{'std::inner_product' is defined in}}
  std::inout_ptr; // expected-error {{no member}} expected-note {{'std::inout_ptr' is defined in}} expected-note {{'std::inout_ptr' is a}}
  std::inout_ptr_t; // expected-error {{no member}} expected-note {{'std::inout_ptr_t' is defined in}} expected-note {{'std::inout_ptr_t' is a}}
  std::inplace_merge; // expected-error {{no member}} expected-note {{'std::inplace_merge' is defined in}}
  std::inplace_vector; // expected-error {{no member}} expected-note {{'std::inplace_vector' is defined in}} expected-note {{'std::inplace_vector' is a}}
  std::input_iterator; // expected-error {{no member}} expected-note {{'std::input_iterator' is defined in}} expected-note {{'std::input_iterator' is a}}
  std::input_iterator_tag; // expected-error {{no member}} expected-note {{'std::input_iterator_tag' is defined in}}
  std::input_or_output_iterator; // expected-error {{no member}} expected-note {{'std::input_or_output_iterator' is defined in}} expected-note {{'std::input_or_output_iterator' is a}}
  std::insert_iterator; // expected-error {{no member}} expected-note {{'std::insert_iterator' is defined in}}
  std::inserter; // expected-error {{no member}} expected-note {{'std::inserter' is defined in}}
  std::int16_t; // expected-error {{no member}} expected-note {{'std::int16_t' is defined in}} expected-note {{'std::int16_t' is a}}
  std::int32_t; // expected-error {{no member}} expected-note {{'std::int32_t' is defined in}} expected-note {{'std::int32_t' is a}}
  std::int64_t; // expected-error {{no member}} expected-note {{'std::int64_t' is defined in}} expected-note {{'std::int64_t' is a}}
  std::int8_t; // expected-error {{no member}} expected-note {{'std::int8_t' is defined in}} expected-note {{'std::int8_t' is a}}
  std::int_fast16_t; // expected-error {{no member}} expected-note {{'std::int_fast16_t' is defined in}} expected-note {{'std::int_fast16_t' is a}}
  std::int_fast32_t; // expected-error {{no member}} expected-note {{'std::int_fast32_t' is defined in}} expected-note {{'std::int_fast32_t' is a}}
  std::int_fast64_t; // expected-error {{no member}} expected-note {{'std::int_fast64_t' is defined in}} expected-note {{'std::int_fast64_t' is a}}
  std::int_fast8_t; // expected-error {{no member}} expected-note {{'std::int_fast8_t' is defined in}} expected-note {{'std::int_fast8_t' is a}}
  std::int_least16_t; // expected-error {{no member}} expected-note {{'std::int_least16_t' is defined in}} expected-note {{'std::int_least16_t' is a}}
  std::int_least32_t; // expected-error {{no member}} expected-note {{'std::int_least32_t' is defined in}} expected-note {{'std::int_least32_t' is a}}
  std::int_least64_t; // expected-error {{no member}} expected-note {{'std::int_least64_t' is defined in}} expected-note {{'std::int_least64_t' is a}}
  std::int_least8_t; // expected-error {{no member}} expected-note {{'std::int_least8_t' is defined in}} expected-note {{'std::int_least8_t' is a}}
  std::integer_sequence; // expected-error {{no member}} expected-note {{'std::integer_sequence' is defined in}} expected-note {{'std::integer_sequence' is a}}
  std::integral; // expected-error {{no member}} expected-note {{'std::integral' is defined in}} expected-note {{'std::integral' is a}}
  std::integral_constant; // expected-error {{no member}} expected-note {{'std::integral_constant' is defined in}} expected-note {{'std::integral_constant' is a}}
  std::internal; // expected-error {{no member}} expected-note {{'std::internal' is defined in}}
  std::internal; // expected-error {{no member}} expected-note {{'std::internal' is defined in}}
  std::intmax_t; // expected-error {{no member}} expected-note {{'std::intmax_t' is defined in}} expected-note {{'std::intmax_t' is a}}
  std::intptr_t; // expected-error {{no member}} expected-note {{'std::intptr_t' is defined in}} expected-note {{'std::intptr_t' is a}}
  std::invalid_argument; // expected-error {{no member}} expected-note {{'std::invalid_argument' is defined in}}
  std::invocable; // expected-error {{no member}} expected-note {{'std::invocable' is defined in}} expected-note {{'std::invocable' is a}}
  std::invoke; // expected-error {{no member}} expected-note {{'std::invoke' is defined in}} expected-note {{'std::invoke' is a}}
  std::invoke_r; // expected-error {{no member}} expected-note {{'std::invoke_r' is defined in}} expected-note {{'std::invoke_r' is a}}
  std::invoke_result; // expected-error {{no member}} expected-note {{'std::invoke_result' is defined in}} expected-note {{'std::invoke_result' is a}}
  std::invoke_result_t; // expected-error {{no member}} expected-note {{'std::invoke_result_t' is defined in}} expected-note {{'std::invoke_result_t' is a}}
  std::io_errc; // expected-error {{no member}} expected-note {{'std::io_errc' is defined in}} expected-note {{'std::io_errc' is a}}
  std::io_errc; // expected-error {{no member}} expected-note {{'std::io_errc' is defined in}} expected-note {{'std::io_errc' is a}}
  std::io_state; // expected-error {{no member}} expected-note {{'std::io_state' is defined in}}
  std::io_state; // expected-error {{no member}} expected-note {{'std::io_state' is defined in}}
  std::ios; // expected-error {{no member}} expected-note {{'std::ios' is defined in}}
  std::ios; // expected-error {{no member}} expected-note {{'std::ios' is defined in}}
  std::ios; // expected-error {{no member}} expected-note {{'std::ios' is defined in}}
  std::ios_base; // expected-error {{no member}} expected-note {{'std::ios_base' is defined in}}
  std::ios_base; // expected-error {{no member}} expected-note {{'std::ios_base' is defined in}}
  std::iostream; // expected-error {{no member}} expected-note {{'std::iostream' is defined in}}
  std::iostream; // expected-error {{no member}} expected-note {{'std::iostream' is defined in}}
  std::iostream; // expected-error {{no member}} expected-note {{'std::iostream' is defined in}}
  std::iostream_category; // expected-error {{no member}} expected-note {{'std::iostream_category' is defined in}} expected-note {{'std::iostream_category' is a}}
  std::iostream_category; // expected-error {{no member}} expected-note {{'std::iostream_category' is defined in}} expected-note {{'std::iostream_category' is a}}
  std::iota; // expected-error {{no member}} expected-note {{'std::iota' is defined in}} expected-note {{'std::iota' is a}}
  std::is_abstract; // expected-error {{no member}} expected-note {{'std::is_abstract' is defined in}} expected-note {{'std::is_abstract' is a}}
  std::is_abstract_v; // expected-error {{no member}} expected-note {{'std::is_abstract_v' is defined in}} expected-note {{'std::is_abstract_v' is a}}
  std::is_aggregate; // expected-error {{no member}} expected-note {{'std::is_aggregate' is defined in}} expected-note {{'std::is_aggregate' is a}}
  std::is_aggregate_v; // expected-error {{no member}} expected-note {{'std::is_aggregate_v' is defined in}} expected-note {{'std::is_aggregate_v' is a}}
  std::is_arithmetic; // expected-error {{no member}} expected-note {{'std::is_arithmetic' is defined in}} expected-note {{'std::is_arithmetic' is a}}
  std::is_arithmetic_v; // expected-error {{no member}} expected-note {{'std::is_arithmetic_v' is defined in}} expected-note {{'std::is_arithmetic_v' is a}}
  std::is_array; // expected-error {{no member}} expected-note {{'std::is_array' is defined in}} expected-note {{'std::is_array' is a}}
  std::is_array_v; // expected-error {{no member}} expected-note {{'std::is_array_v' is defined in}} expected-note {{'std::is_array_v' is a}}
  std::is_assignable; // expected-error {{no member}} expected-note {{'std::is_assignable' is defined in}} expected-note {{'std::is_assignable' is a}}
  std::is_assignable_v; // expected-error {{no member}} expected-note {{'std::is_assignable_v' is defined in}} expected-note {{'std::is_assignable_v' is a}}
  std::is_base_of; // expected-error {{no member}} expected-note {{'std::is_base_of' is defined in}} expected-note {{'std::is_base_of' is a}}
  std::is_base_of_v; // expected-error {{no member}} expected-note {{'std::is_base_of_v' is defined in}} expected-note {{'std::is_base_of_v' is a}}
  std::is_bind_expression; // expected-error {{no member}} expected-note {{'std::is_bind_expression' is defined in}} expected-note {{'std::is_bind_expression' is a}}
  std::is_bind_expression_v; // expected-error {{no member}} expected-note {{'std::is_bind_expression_v' is defined in}} expected-note {{'std::is_bind_expression_v' is a}}
  std::is_bounded_array; // expected-error {{no member}} expected-note {{'std::is_bounded_array' is defined in}} expected-note {{'std::is_bounded_array' is a}}
  std::is_bounded_array_v; // expected-error {{no member}} expected-note {{'std::is_bounded_array_v' is defined in}} expected-note {{'std::is_bounded_array_v' is a}}
  std::is_class; // expected-error {{no member}} expected-note {{'std::is_class' is defined in}} expected-note {{'std::is_class' is a}}
  std::is_class_v; // expected-error {{no member}} expected-note {{'std::is_class_v' is defined in}} expected-note {{'std::is_class_v' is a}}
  std::is_compound; // expected-error {{no member}} expected-note {{'std::is_compound' is defined in}} expected-note {{'std::is_compound' is a}}
  std::is_compound_v; // expected-error {{no member}} expected-note {{'std::is_compound_v' is defined in}} expected-note {{'std::is_compound_v' is a}}
  std::is_const; // expected-error {{no member}} expected-note {{'std::is_const' is defined in}} expected-note {{'std::is_const' is a}}
  std::is_const_v; // expected-error {{no member}} expected-note {{'std::is_const_v' is defined in}} expected-note {{'std::is_const_v' is a}}
  std::is_constant_evaluated; // expected-error {{no member}} expected-note {{'std::is_constant_evaluated' is defined in}} expected-note {{'std::is_constant_evaluated' is a}}
  std::is_constructible; // expected-error {{no member}} expected-note {{'std::is_constructible' is defined in}} expected-note {{'std::is_constructible' is a}}
  std::is_constructible_v; // expected-error {{no member}} expected-note {{'std::is_constructible_v' is defined in}} expected-note {{'std::is_constructible_v' is a}}
  std::is_convertible; // expected-error {{no member}} expected-note {{'std::is_convertible' is defined in}} expected-note {{'std::is_convertible' is a}}
  std::is_convertible_v; // expected-error {{no member}} expected-note {{'std::is_convertible_v' is defined in}} expected-note {{'std::is_convertible_v' is a}}
  std::is_copy_assignable; // expected-error {{no member}} expected-note {{'std::is_copy_assignable' is defined in}} expected-note {{'std::is_copy_assignable' is a}}
  std::is_copy_assignable_v; // expected-error {{no member}} expected-note {{'std::is_copy_assignable_v' is defined in}} expected-note {{'std::is_copy_assignable_v' is a}}
  std::is_copy_constructible; // expected-error {{no member}} expected-note {{'std::is_copy_constructible' is defined in}} expected-note {{'std::is_copy_constructible' is a}}
  std::is_copy_constructible_v; // expected-error {{no member}} expected-note {{'std::is_copy_constructible_v' is defined in}} expected-note {{'std::is_copy_constructible_v' is a}}
  std::is_corresponding_member; // expected-error {{no member}} expected-note {{'std::is_corresponding_member' is defined in}} expected-note {{'std::is_corresponding_member' is a}}
  std::is_debugger_present; // expected-error {{no member}} expected-note {{'std::is_debugger_present' is defined in}} expected-note {{'std::is_debugger_present' is a}}
  std::is_default_constructible; // expected-error {{no member}} expected-note {{'std::is_default_constructible' is defined in}} expected-note {{'std::is_default_constructible' is a}}
  std::is_default_constructible_v; // expected-error {{no member}} expected-note {{'std::is_default_constructible_v' is defined in}} expected-note {{'std::is_default_constructible_v' is a}}
  std::is_destructible; // expected-error {{no member}} expected-note {{'std::is_destructible' is defined in}} expected-note {{'std::is_destructible' is a}}
  std::is_destructible_v; // expected-error {{no member}} expected-note {{'std::is_destructible_v' is defined in}} expected-note {{'std::is_destructible_v' is a}}
  std::is_empty; // expected-error {{no member}} expected-note {{'std::is_empty' is defined in}} expected-note {{'std::is_empty' is a}}
  std::is_empty_v; // expected-error {{no member}} expected-note {{'std::is_empty_v' is defined in}} expected-note {{'std::is_empty_v' is a}}
  std::is_enum; // expected-error {{no member}} expected-note {{'std::is_enum' is defined in}} expected-note {{'std::is_enum' is a}}
  std::is_enum_v; // expected-error {{no member}} expected-note {{'std::is_enum_v' is defined in}} expected-note {{'std::is_enum_v' is a}}
  std::is_eq; // expected-error {{no member}} expected-note {{'std::is_eq' is defined in}} expected-note {{'std::is_eq' is a}}
  std::is_error_code_enum; // expected-error {{no member}} expected-note {{'std::is_error_code_enum' is defined in}} expected-note {{'std::is_error_code_enum' is a}}
  std::is_error_condition_enum; // expected-error {{no member}} expected-note {{'std::is_error_condition_enum' is defined in}} expected-note {{'std::is_error_condition_enum' is a}}
  std::is_error_condition_enum_v; // expected-error {{no member}} expected-note {{'std::is_error_condition_enum_v' is defined in}} expected-note {{'std::is_error_condition_enum_v' is a}}
  std::is_execution_policy; // expected-error {{no member}} expected-note {{'std::is_execution_policy' is defined in}} expected-note {{'std::is_execution_policy' is a}}
  std::is_execution_policy_v; // expected-error {{no member}} expected-note {{'std::is_execution_policy_v' is defined in}} expected-note {{'std::is_execution_policy_v' is a}}
  std::is_final; // expected-error {{no member}} expected-note {{'std::is_final' is defined in}} expected-note {{'std::is_final' is a}}
  std::is_final_v; // expected-error {{no member}} expected-note {{'std::is_final_v' is defined in}} expected-note {{'std::is_final_v' is a}}
  std::is_floating_point; // expected-error {{no member}} expected-note {{'std::is_floating_point' is defined in}} expected-note {{'std::is_floating_point' is a}}
  std::is_floating_point_v; // expected-error {{no member}} expected-note {{'std::is_floating_point_v' is defined in}} expected-note {{'std::is_floating_point_v' is a}}
  std::is_function; // expected-error {{no member}} expected-note {{'std::is_function' is defined in}} expected-note {{'std::is_function' is a}}
  std::is_function_v; // expected-error {{no member}} expected-note {{'std::is_function_v' is defined in}} expected-note {{'std::is_function_v' is a}}
  std::is_fundamental; // expected-error {{no member}} expected-note {{'std::is_fundamental' is defined in}} expected-note {{'std::is_fundamental' is a}}
  std::is_fundamental_v; // expected-error {{no member}} expected-note {{'std::is_fundamental_v' is defined in}} expected-note {{'std::is_fundamental_v' is a}}
  std::is_gt; // expected-error {{no member}} expected-note {{'std::is_gt' is defined in}} expected-note {{'std::is_gt' is a}}
  std::is_gteq; // expected-error {{no member}} expected-note {{'std::is_gteq' is defined in}} expected-note {{'std::is_gteq' is a}}
  std::is_heap; // expected-error {{no member}} expected-note {{'std::is_heap' is defined in}} expected-note {{'std::is_heap' is a}}
  std::is_heap_until; // expected-error {{no member}} expected-note {{'std::is_heap_until' is defined in}} expected-note {{'std::is_heap_until' is a}}
  std::is_implicit_lifetime; // expected-error {{no member}} expected-note {{'std::is_implicit_lifetime' is defined in}} expected-note {{'std::is_implicit_lifetime' is a}}
  std::is_integral; // expected-error {{no member}} expected-note {{'std::is_integral' is defined in}} expected-note {{'std::is_integral' is a}}
  std::is_integral_v; // expected-error {{no member}} expected-note {{'std::is_integral_v' is defined in}} expected-note {{'std::is_integral_v' is a}}
  std::is_invocable; // expected-error {{no member}} expected-note {{'std::is_invocable' is defined in}} expected-note {{'std::is_invocable' is a}}
  std::is_invocable_r; // expected-error {{no member}} expected-note {{'std::is_invocable_r' is defined in}} expected-note {{'std::is_invocable_r' is a}}
  std::is_invocable_r_v; // expected-error {{no member}} expected-note {{'std::is_invocable_r_v' is defined in}} expected-note {{'std::is_invocable_r_v' is a}}
  std::is_invocable_v; // expected-error {{no member}} expected-note {{'std::is_invocable_v' is defined in}} expected-note {{'std::is_invocable_v' is a}}
  std::is_layout_compatible; // expected-error {{no member}} expected-note {{'std::is_layout_compatible' is defined in}} expected-note {{'std::is_layout_compatible' is a}}
  std::is_layout_compatible_v; // expected-error {{no member}} expected-note {{'std::is_layout_compatible_v' is defined in}} expected-note {{'std::is_layout_compatible_v' is a}}
  std::is_literal_type; // expected-error {{no member}} expected-note {{'std::is_literal_type' is defined in}} expected-note {{'std::is_literal_type' is a}}
  std::is_literal_type_v; // expected-error {{no member}} expected-note {{'std::is_literal_type_v' is defined in}} expected-note {{'std::is_literal_type_v' is a}}
  std::is_lt; // expected-error {{no member}} expected-note {{'std::is_lt' is defined in}} expected-note {{'std::is_lt' is a}}
  std::is_lteq; // expected-error {{no member}} expected-note {{'std::is_lteq' is defined in}} expected-note {{'std::is_lteq' is a}}
  std::is_lvalue_reference; // expected-error {{no member}} expected-note {{'std::is_lvalue_reference' is defined in}} expected-note {{'std::is_lvalue_reference' is a}}
  std::is_lvalue_reference_v; // expected-error {{no member}} expected-note {{'std::is_lvalue_reference_v' is defined in}} expected-note {{'std::is_lvalue_reference_v' is a}}
  std::is_member_function_pointer; // expected-error {{no member}} expected-note {{'std::is_member_function_pointer' is defined in}} expected-note {{'std::is_member_function_pointer' is a}}
  std::is_member_function_pointer_v; // expected-error {{no member}} expected-note {{'std::is_member_function_pointer_v' is defined in}} expected-note {{'std::is_member_function_pointer_v' is a}}
  std::is_member_object_pointer; // expected-error {{no member}} expected-note {{'std::is_member_object_pointer' is defined in}} expected-note {{'std::is_member_object_pointer' is a}}
  std::is_member_object_pointer_v; // expected-error {{no member}} expected-note {{'std::is_member_object_pointer_v' is defined in}} expected-note {{'std::is_member_object_pointer_v' is a}}
  std::is_member_pointer; // expected-error {{no member}} expected-note {{'std::is_member_pointer' is defined in}} expected-note {{'std::is_member_pointer' is a}}
  std::is_member_pointer_v; // expected-error {{no member}} expected-note {{'std::is_member_pointer_v' is defined in}} expected-note {{'std::is_member_pointer_v' is a}}
  std::is_move_assignable; // expected-error {{no member}} expected-note {{'std::is_move_assignable' is defined in}} expected-note {{'std::is_move_assignable' is a}}
  std::is_move_assignable_v; // expected-error {{no member}} expected-note {{'std::is_move_assignable_v' is defined in}} expected-note {{'std::is_move_assignable_v' is a}}
  std::is_move_constructible; // expected-error {{no member}} expected-note {{'std::is_move_constructible' is defined in}} expected-note {{'std::is_move_constructible' is a}}
  std::is_move_constructible_v; // expected-error {{no member}} expected-note {{'std::is_move_constructible_v' is defined in}} expected-note {{'std::is_move_constructible_v' is a}}
  std::is_neq; // expected-error {{no member}} expected-note {{'std::is_neq' is defined in}} expected-note {{'std::is_neq' is a}}
  std::is_nothrow_assignable; // expected-error {{no member}} expected-note {{'std::is_nothrow_assignable' is defined in}} expected-note {{'std::is_nothrow_assignable' is a}}
  std::is_nothrow_assignable_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_assignable_v' is defined in}} expected-note {{'std::is_nothrow_assignable_v' is a}}
  std::is_nothrow_constructible; // expected-error {{no member}} expected-note {{'std::is_nothrow_constructible' is defined in}} expected-note {{'std::is_nothrow_constructible' is a}}
  std::is_nothrow_constructible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_constructible_v' is defined in}} expected-note {{'std::is_nothrow_constructible_v' is a}}
  std::is_nothrow_convertible; // expected-error {{no member}} expected-note {{'std::is_nothrow_convertible' is defined in}} expected-note {{'std::is_nothrow_convertible' is a}}
  std::is_nothrow_convertible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_convertible_v' is defined in}} expected-note {{'std::is_nothrow_convertible_v' is a}}
  std::is_nothrow_copy_assignable; // expected-error {{no member}} expected-note {{'std::is_nothrow_copy_assignable' is defined in}} expected-note {{'std::is_nothrow_copy_assignable' is a}}
  std::is_nothrow_copy_assignable_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_copy_assignable_v' is defined in}} expected-note {{'std::is_nothrow_copy_assignable_v' is a}}
  std::is_nothrow_copy_constructible; // expected-error {{no member}} expected-note {{'std::is_nothrow_copy_constructible' is defined in}} expected-note {{'std::is_nothrow_copy_constructible' is a}}
  std::is_nothrow_copy_constructible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_copy_constructible_v' is defined in}} expected-note {{'std::is_nothrow_copy_constructible_v' is a}}
  std::is_nothrow_default_constructible; // expected-error {{no member}} expected-note {{'std::is_nothrow_default_constructible' is defined in}} expected-note {{'std::is_nothrow_default_constructible' is a}}
  std::is_nothrow_default_constructible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_default_constructible_v' is defined in}} expected-note {{'std::is_nothrow_default_constructible_v' is a}}
  std::is_nothrow_destructible; // expected-error {{no member}} expected-note {{'std::is_nothrow_destructible' is defined in}} expected-note {{'std::is_nothrow_destructible' is a}}
  std::is_nothrow_destructible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_destructible_v' is defined in}} expected-note {{'std::is_nothrow_destructible_v' is a}}
  std::is_nothrow_invocable; // expected-error {{no member}} expected-note {{'std::is_nothrow_invocable' is defined in}} expected-note {{'std::is_nothrow_invocable' is a}}
  std::is_nothrow_invocable_r; // expected-error {{no member}} expected-note {{'std::is_nothrow_invocable_r' is defined in}} expected-note {{'std::is_nothrow_invocable_r' is a}}
  std::is_nothrow_invocable_r_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_invocable_r_v' is defined in}} expected-note {{'std::is_nothrow_invocable_r_v' is a}}
  std::is_nothrow_invocable_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_invocable_v' is defined in}} expected-note {{'std::is_nothrow_invocable_v' is a}}
  std::is_nothrow_move_assignable; // expected-error {{no member}} expected-note {{'std::is_nothrow_move_assignable' is defined in}} expected-note {{'std::is_nothrow_move_assignable' is a}}
  std::is_nothrow_move_assignable_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_move_assignable_v' is defined in}} expected-note {{'std::is_nothrow_move_assignable_v' is a}}
  std::is_nothrow_move_constructible; // expected-error {{no member}} expected-note {{'std::is_nothrow_move_constructible' is defined in}} expected-note {{'std::is_nothrow_move_constructible' is a}}
  std::is_nothrow_move_constructible_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_move_constructible_v' is defined in}} expected-note {{'std::is_nothrow_move_constructible_v' is a}}
  std::is_nothrow_swappable; // expected-error {{no member}} expected-note {{'std::is_nothrow_swappable' is defined in}} expected-note {{'std::is_nothrow_swappable' is a}}
  std::is_nothrow_swappable_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_swappable_v' is defined in}} expected-note {{'std::is_nothrow_swappable_v' is a}}
  std::is_nothrow_swappable_with; // expected-error {{no member}} expected-note {{'std::is_nothrow_swappable_with' is defined in}} expected-note {{'std::is_nothrow_swappable_with' is a}}
  std::is_nothrow_swappable_with_v; // expected-error {{no member}} expected-note {{'std::is_nothrow_swappable_with_v' is defined in}} expected-note {{'std::is_nothrow_swappable_with_v' is a}}
  std::is_null_pointer; // expected-error {{no member}} expected-note {{'std::is_null_pointer' is defined in}} expected-note {{'std::is_null_pointer' is a}}
  std::is_null_pointer_v; // expected-error {{no member}} expected-note {{'std::is_null_pointer_v' is defined in}} expected-note {{'std::is_null_pointer_v' is a}}
  std::is_object; // expected-error {{no member}} expected-note {{'std::is_object' is defined in}} expected-note {{'std::is_object' is a}}
  std::is_object_v; // expected-error {{no member}} expected-note {{'std::is_object_v' is defined in}} expected-note {{'std::is_object_v' is a}}
  std::is_partitioned; // expected-error {{no member}} expected-note {{'std::is_partitioned' is defined in}} expected-note {{'std::is_partitioned' is a}}
  std::is_permutation; // expected-error {{no member}} expected-note {{'std::is_permutation' is defined in}} expected-note {{'std::is_permutation' is a}}
  std::is_placeholder; // expected-error {{no member}} expected-note {{'std::is_placeholder' is defined in}} expected-note {{'std::is_placeholder' is a}}
  std::is_placeholder_v; // expected-error {{no member}} expected-note {{'std::is_placeholder_v' is defined in}} expected-note {{'std::is_placeholder_v' is a}}
  std::is_pod; // expected-error {{no member}} expected-note {{'std::is_pod' is defined in}} expected-note {{'std::is_pod' is a}}
  std::is_pod_v; // expected-error {{no member}} expected-note {{'std::is_pod_v' is defined in}} expected-note {{'std::is_pod_v' is a}}
  std::is_pointer; // expected-error {{no member}} expected-note {{'std::is_pointer' is defined in}} expected-note {{'std::is_pointer' is a}}
  std::is_pointer_interconvertible_base_of; // expected-error {{no member}} expected-note {{'std::is_pointer_interconvertible_base_of' is defined in}} expected-note {{'std::is_pointer_interconvertible_base_of' is a}}
  std::is_pointer_interconvertible_base_of_v; // expected-error {{no member}} expected-note {{'std::is_pointer_interconvertible_base_of_v' is defined in}} expected-note {{'std::is_pointer_interconvertible_base_of_v' is a}}
  std::is_pointer_interconvertible_with_class; // expected-error {{no member}} expected-note {{'std::is_pointer_interconvertible_with_class' is defined in}} expected-note {{'std::is_pointer_interconvertible_with_class' is a}}
  std::is_pointer_v; // expected-error {{no member}} expected-note {{'std::is_pointer_v' is defined in}} expected-note {{'std::is_pointer_v' is a}}
  std::is_polymorphic; // expected-error {{no member}} expected-note {{'std::is_polymorphic' is defined in}} expected-note {{'std::is_polymorphic' is a}}
  std::is_polymorphic_v; // expected-error {{no member}} expected-note {{'std::is_polymorphic_v' is defined in}} expected-note {{'std::is_polymorphic_v' is a}}
  std::is_reference; // expected-error {{no member}} expected-note {{'std::is_reference' is defined in}} expected-note {{'std::is_reference' is a}}
  std::is_reference_v; // expected-error {{no member}} expected-note {{'std::is_reference_v' is defined in}} expected-note {{'std::is_reference_v' is a}}
  std::is_rvalue_reference; // expected-error {{no member}} expected-note {{'std::is_rvalue_reference' is defined in}} expected-note {{'std::is_rvalue_reference' is a}}
  std::is_rvalue_reference_v; // expected-error {{no member}} expected-note {{'std::is_rvalue_reference_v' is defined in}} expected-note {{'std::is_rvalue_reference_v' is a}}
  std::is_same; // expected-error {{no member}} expected-note {{'std::is_same' is defined in}} expected-note {{'std::is_same' is a}}
  std::is_same_v; // expected-error {{no member}} expected-note {{'std::is_same_v' is defined in}} expected-note {{'std::is_same_v' is a}}
  std::is_scalar; // expected-error {{no member}} expected-note {{'std::is_scalar' is defined in}} expected-note {{'std::is_scalar' is a}}
  std::is_scalar_v; // expected-error {{no member}} expected-note {{'std::is_scalar_v' is defined in}} expected-note {{'std::is_scalar_v' is a}}
  std::is_scoped_enum; // expected-error {{no member}} expected-note {{'std::is_scoped_enum' is defined in}} expected-note {{'std::is_scoped_enum' is a}}
  std::is_scoped_enum_v; // expected-error {{no member}} expected-note {{'std::is_scoped_enum_v' is defined in}} expected-note {{'std::is_scoped_enum_v' is a}}
  std::is_signed; // expected-error {{no member}} expected-note {{'std::is_signed' is defined in}} expected-note {{'std::is_signed' is a}}
  std::is_signed_v; // expected-error {{no member}} expected-note {{'std::is_signed_v' is defined in}} expected-note {{'std::is_signed_v' is a}}
  std::is_sorted; // expected-error {{no member}} expected-note {{'std::is_sorted' is defined in}} expected-note {{'std::is_sorted' is a}}
  std::is_sorted_until; // expected-error {{no member}} expected-note {{'std::is_sorted_until' is defined in}} expected-note {{'std::is_sorted_until' is a}}
  std::is_standard_layout; // expected-error {{no member}} expected-note {{'std::is_standard_layout' is defined in}} expected-note {{'std::is_standard_layout' is a}}
  std::is_standard_layout_v; // expected-error {{no member}} expected-note {{'std::is_standard_layout_v' is defined in}} expected-note {{'std::is_standard_layout_v' is a}}
  std::is_swappable; // expected-error {{no member}} expected-note {{'std::is_swappable' is defined in}} expected-note {{'std::is_swappable' is a}}
  std::is_swappable_v; // expected-error {{no member}} expected-note {{'std::is_swappable_v' is defined in}} expected-note {{'std::is_swappable_v' is a}}
  std::is_swappable_with; // expected-error {{no member}} expected-note {{'std::is_swappable_with' is defined in}} expected-note {{'std::is_swappable_with' is a}}
  std::is_swappable_with_v; // expected-error {{no member}} expected-note {{'std::is_swappable_with_v' is defined in}} expected-note {{'std::is_swappable_with_v' is a}}
  std::is_trivial; // expected-error {{no member}} expected-note {{'std::is_trivial' is defined in}} expected-note {{'std::is_trivial' is a}}
  std::is_trivial_v; // expected-error {{no member}} expected-note {{'std::is_trivial_v' is defined in}} expected-note {{'std::is_trivial_v' is a}}
  std::is_trivially_assignable; // expected-error {{no member}} expected-note {{'std::is_trivially_assignable' is defined in}} expected-note {{'std::is_trivially_assignable' is a}}
  std::is_trivially_assignable_v; // expected-error {{no member}} expected-note {{'std::is_trivially_assignable_v' is defined in}} expected-note {{'std::is_trivially_assignable_v' is a}}
  std::is_trivially_constructible; // expected-error {{no member}} expected-note {{'std::is_trivially_constructible' is defined in}} expected-note {{'std::is_trivially_constructible' is a}}
  std::is_trivially_constructible_v; // expected-error {{no member}} expected-note {{'std::is_trivially_constructible_v' is defined in}} expected-note {{'std::is_trivially_constructible_v' is a}}
  std::is_trivially_copy_assignable; // expected-error {{no member}} expected-note {{'std::is_trivially_copy_assignable' is defined in}} expected-note {{'std::is_trivially_copy_assignable' is a}}
  std::is_trivially_copy_assignable_v; // expected-error {{no member}} expected-note {{'std::is_trivially_copy_assignable_v' is defined in}} expected-note {{'std::is_trivially_copy_assignable_v' is a}}
  std::is_trivially_copy_constructible; // expected-error {{no member}} expected-note {{'std::is_trivially_copy_constructible' is defined in}} expected-note {{'std::is_trivially_copy_constructible' is a}}
  std::is_trivially_copy_constructible_v; // expected-error {{no member}} expected-note {{'std::is_trivially_copy_constructible_v' is defined in}} expected-note {{'std::is_trivially_copy_constructible_v' is a}}
  std::is_trivially_copyable; // expected-error {{no member}} expected-note {{'std::is_trivially_copyable' is defined in}} expected-note {{'std::is_trivially_copyable' is a}}
  std::is_trivially_copyable_v; // expected-error {{no member}} expected-note {{'std::is_trivially_copyable_v' is defined in}} expected-note {{'std::is_trivially_copyable_v' is a}}
  std::is_trivially_default_constructible; // expected-error {{no member}} expected-note {{'std::is_trivially_default_constructible' is defined in}} expected-note {{'std::is_trivially_default_constructible' is a}}
  std::is_trivially_default_constructible_v; // expected-error {{no member}} expected-note {{'std::is_trivially_default_constructible_v' is defined in}} expected-note {{'std::is_trivially_default_constructible_v' is a}}
  std::is_trivially_destructible; // expected-error {{no member}} expected-note {{'std::is_trivially_destructible' is defined in}} expected-note {{'std::is_trivially_destructible' is a}}
  std::is_trivially_destructible_v; // expected-error {{no member}} expected-note {{'std::is_trivially_destructible_v' is defined in}} expected-note {{'std::is_trivially_destructible_v' is a}}
  std::is_trivially_move_assignable; // expected-error {{no member}} expected-note {{'std::is_trivially_move_assignable' is defined in}} expected-note {{'std::is_trivially_move_assignable' is a}}
  std::is_trivially_move_assignable_v; // expected-error {{no member}} expected-note {{'std::is_trivially_move_assignable_v' is defined in}} expected-note {{'std::is_trivially_move_assignable_v' is a}}
  std::is_trivially_move_constructible; // expected-error {{no member}} expected-note {{'std::is_trivially_move_constructible' is defined in}} expected-note {{'std::is_trivially_move_constructible' is a}}
  std::is_trivially_move_constructible_v; // expected-error {{no member}} expected-note {{'std::is_trivially_move_constructible_v' is defined in}} expected-note {{'std::is_trivially_move_constructible_v' is a}}
  std::is_unbounded_array; // expected-error {{no member}} expected-note {{'std::is_unbounded_array' is defined in}} expected-note {{'std::is_unbounded_array' is a}}
  std::is_unbounded_array_v; // expected-error {{no member}} expected-note {{'std::is_unbounded_array_v' is defined in}} expected-note {{'std::is_unbounded_array_v' is a}}
  std::is_union; // expected-error {{no member}} expected-note {{'std::is_union' is defined in}} expected-note {{'std::is_union' is a}}
  std::is_union_v; // expected-error {{no member}} expected-note {{'std::is_union_v' is defined in}} expected-note {{'std::is_union_v' is a}}
  std::is_unsigned; // expected-error {{no member}} expected-note {{'std::is_unsigned' is defined in}} expected-note {{'std::is_unsigned' is a}}
  std::is_unsigned_v; // expected-error {{no member}} expected-note {{'std::is_unsigned_v' is defined in}} expected-note {{'std::is_unsigned_v' is a}}
  std::is_virtual_base_of; // expected-error {{no member}} expected-note {{'std::is_virtual_base_of' is defined in}} expected-note {{'std::is_virtual_base_of' is a}}
  std::is_virtual_base_of_v; // expected-error {{no member}} expected-note {{'std::is_virtual_base_of_v' is defined in}} expected-note {{'std::is_virtual_base_of_v' is a}}
  std::is_void; // expected-error {{no member}} expected-note {{'std::is_void' is defined in}} expected-note {{'std::is_void' is a}}
  std::is_void_v; // expected-error {{no member}} expected-note {{'std::is_void_v' is defined in}} expected-note {{'std::is_void_v' is a}}
  std::is_volatile; // expected-error {{no member}} expected-note {{'std::is_volatile' is defined in}} expected-note {{'std::is_volatile' is a}}
  std::is_volatile_v; // expected-error {{no member}} expected-note {{'std::is_volatile_v' is defined in}} expected-note {{'std::is_volatile_v' is a}}
  std::is_within_lifetime; // expected-error {{no member}} expected-note {{'std::is_within_lifetime' is defined in}} expected-note {{'std::is_within_lifetime' is a}}
  std::isalnum; // expected-error {{no member}} expected-note {{'std::isalnum' is defined in}}
  std::isalpha; // expected-error {{no member}} expected-note {{'std::isalpha' is defined in}}
  std::isblank; // expected-error {{no member}} expected-note {{'std::isblank' is defined in}} expected-note {{'std::isblank' is a}}
  std::iscntrl; // expected-error {{no member}} expected-note {{'std::iscntrl' is defined in}}
  std::isdigit; // expected-error {{no member}} expected-note {{'std::isdigit' is defined in}}
  std::isgraph; // expected-error {{no member}} expected-note {{'std::isgraph' is defined in}}
  std::isgreater; // expected-error {{no member}} expected-note {{'std::isgreater' is defined in}} expected-note {{'std::isgreater' is a}}
  std::isgreaterequal; // expected-error {{no member}} expected-note {{'std::isgreaterequal' is defined in}} expected-note {{'std::isgreaterequal' is a}}
  std::isless; // expected-error {{no member}} expected-note {{'std::isless' is defined in}} expected-note {{'std::isless' is a}}
  std::islessequal; // expected-error {{no member}} expected-note {{'std::islessequal' is defined in}} expected-note {{'std::islessequal' is a}}
  std::islessgreater; // expected-error {{no member}} expected-note {{'std::islessgreater' is defined in}} expected-note {{'std::islessgreater' is a}}
  std::islower; // expected-error {{no member}} expected-note {{'std::islower' is defined in}}
  std::ispanstream; // expected-error {{no member}} expected-note {{'std::ispanstream' is defined in}} expected-note {{'std::ispanstream' is a}}
  std::ispanstream; // expected-error {{no member}} expected-note {{'std::ispanstream' is defined in}} expected-note {{'std::ispanstream' is a}}
  std::isprint; // expected-error {{no member}} expected-note {{'std::isprint' is defined in}}
  std::ispunct; // expected-error {{no member}} expected-note {{'std::ispunct' is defined in}}
  std::isspace; // expected-error {{no member}} expected-note {{'std::isspace' is defined in}}
  std::istream; // expected-error {{no member}} expected-note {{'std::istream' is defined in}}
  std::istream; // expected-error {{no member}} expected-note {{'std::istream' is defined in}}
  std::istream; // expected-error {{no member}} expected-note {{'std::istream' is defined in}}
  std::istream_iterator; // expected-error {{no member}} expected-note {{'std::istream_iterator' is defined in}}
  std::istreambuf_iterator; // expected-error {{no member}} expected-note {{'std::istreambuf_iterator' is defined in}}
  std::istreambuf_iterator; // expected-error {{no member}} expected-note {{'std::istreambuf_iterator' is defined in}}
  std::istringstream; // expected-error {{no member}} expected-note {{'std::istringstream' is defined in}}
  std::istringstream; // expected-error {{no member}} expected-note {{'std::istringstream' is defined in}}
  std::istrstream; // expected-error {{no member}} expected-note {{'std::istrstream' is defined in}}
  std::isunordered; // expected-error {{no member}} expected-note {{'std::isunordered' is defined in}} expected-note {{'std::isunordered' is a}}
  std::isupper; // expected-error {{no member}} expected-note {{'std::isupper' is defined in}}
  std::iswalnum; // expected-error {{no member}} expected-note {{'std::iswalnum' is defined in}}
  std::iswalpha; // expected-error {{no member}} expected-note {{'std::iswalpha' is defined in}}
  std::iswblank; // expected-error {{no member}} expected-note {{'std::iswblank' is defined in}} expected-note {{'std::iswblank' is a}}
  std::iswcntrl; // expected-error {{no member}} expected-note {{'std::iswcntrl' is defined in}}
  std::iswctype; // expected-error {{no member}} expected-note {{'std::iswctype' is defined in}}
  std::iswdigit; // expected-error {{no member}} expected-note {{'std::iswdigit' is defined in}}
  std::iswgraph; // expected-error {{no member}} expected-note {{'std::iswgraph' is defined in}}
  std::iswlower; // expected-error {{no member}} expected-note {{'std::iswlower' is defined in}}
  std::iswprint; // expected-error {{no member}} expected-note {{'std::iswprint' is defined in}}
  std::iswpunct; // expected-error {{no member}} expected-note {{'std::iswpunct' is defined in}}
  std::iswspace; // expected-error {{no member}} expected-note {{'std::iswspace' is defined in}}
  std::iswupper; // expected-error {{no member}} expected-note {{'std::iswupper' is defined in}}
  std::iswxdigit; // expected-error {{no member}} expected-note {{'std::iswxdigit' is defined in}}
  std::isxdigit; // expected-error {{no member}} expected-note {{'std::isxdigit' is defined in}}
  std::iter_common_reference_t; // expected-error {{no member}} expected-note {{'std::iter_common_reference_t' is defined in}} expected-note {{'std::iter_common_reference_t' is a}}
  std::iter_const_reference_t; // expected-error {{no member}} expected-note {{'std::iter_const_reference_t' is defined in}} expected-note {{'std::iter_const_reference_t' is a}}
  std::iter_difference_t; // expected-error {{no member}} expected-note {{'std::iter_difference_t' is defined in}} expected-note {{'std::iter_difference_t' is a}}
  std::iter_reference_t; // expected-error {{no member}} expected-note {{'std::iter_reference_t' is defined in}} expected-note {{'std::iter_reference_t' is a}}
  std::iter_rvalue_reference_t; // expected-error {{no member}} expected-note {{'std::iter_rvalue_reference_t' is defined in}} expected-note {{'std::iter_rvalue_reference_t' is a}}
  std::iter_swap; // expected-error {{no member}} expected-note {{'std::iter_swap' is defined in}}
  std::iter_value_t; // expected-error {{no member}} expected-note {{'std::iter_value_t' is defined in}} expected-note {{'std::iter_value_t' is a}}
  std::iterator; // expected-error {{no member}} expected-note {{'std::iterator' is defined in}}
  std::iterator_traits; // expected-error {{no member}} expected-note {{'std::iterator_traits' is defined in}}
  std::jmp_buf; // expected-error {{no member}} expected-note {{'std::jmp_buf' is defined in}}
  std::jthread; // expected-error {{no member}} expected-note {{'std::jthread' is defined in}} expected-note {{'std::jthread' is a}}
  std::kill_dependency; // expected-error {{no member}} expected-note {{'std::kill_dependency' is defined in}} expected-note {{'std::kill_dependency' is a}}
  std::kilo; // expected-error {{no member}} expected-note {{'std::kilo' is defined in}} expected-note {{'std::kilo' is a}}
  std::knuth_b; // expected-error {{no member}} expected-note {{'std::knuth_b' is defined in}} expected-note {{'std::knuth_b' is a}}
  std::labs; // expected-error {{no member}} expected-note {{'std::labs' is defined in}}
  std::laguerre; // expected-error {{no member}} expected-note {{'std::laguerre' is defined in}} expected-note {{'std::laguerre' is a}}
  std::laguerref; // expected-error {{no member}} expected-note {{'std::laguerref' is defined in}} expected-note {{'std::laguerref' is a}}
  std::laguerrel; // expected-error {{no member}} expected-note {{'std::laguerrel' is defined in}} expected-note {{'std::laguerrel' is a}}
  std::latch; // expected-error {{no member}} expected-note {{'std::latch' is defined in}} expected-note {{'std::latch' is a}}
  std::launch; // expected-error {{no member}} expected-note {{'std::launch' is defined in}} expected-note {{'std::launch' is a}}
  std::launder; // expected-error {{no member}} expected-note {{'std::launder' is defined in}} expected-note {{'std::launder' is a}}
  std::layout_left; // expected-error {{no member}} expected-note {{'std::layout_left' is defined in}} expected-note {{'std::layout_left' is a}}
  std::layout_left_padded; // expected-error {{no member}} expected-note {{'std::layout_left_padded' is defined in}} expected-note {{'std::layout_left_padded' is a}}
  std::layout_right; // expected-error {{no member}} expected-note {{'std::layout_right' is defined in}} expected-note {{'std::layout_right' is a}}
  std::layout_right_padded; // expected-error {{no member}} expected-note {{'std::layout_right_padded' is defined in}} expected-note {{'std::layout_right_padded' is a}}
  std::layout_stride; // expected-error {{no member}} expected-note {{'std::layout_stride' is defined in}} expected-note {{'std::layout_stride' is a}}
  std::lcm; // expected-error {{no member}} expected-note {{'std::lcm' is defined in}} expected-note {{'std::lcm' is a}}
  std::lconv; // expected-error {{no member}} expected-note {{'std::lconv' is defined in}}
  std::ldexp; // expected-error {{no member}} expected-note {{'std::ldexp' is defined in}}
  std::ldexpf; // expected-error {{no member}} expected-note {{'std::ldexpf' is defined in}} expected-note {{'std::ldexpf' is a}}
  std::ldexpl; // expected-error {{no member}} expected-note {{'std::ldexpl' is defined in}} expected-note {{'std::ldexpl' is a}}
  std::ldiv; // expected-error {{no member}} expected-note {{'std::ldiv' is defined in}}
  std::ldiv_t; // expected-error {{no member}} expected-note {{'std::ldiv_t' is defined in}}
  std::left; // expected-error {{no member}} expected-note {{'std::left' is defined in}}
  std::left; // expected-error {{no member}} expected-note {{'std::left' is defined in}}
  std::legendre; // expected-error {{no member}} expected-note {{'std::legendre' is defined in}} expected-note {{'std::legendre' is a}}
  std::legendref; // expected-error {{no member}} expected-note {{'std::legendref' is defined in}} expected-note {{'std::legendref' is a}}
  std::legendrel; // expected-error {{no member}} expected-note {{'std::legendrel' is defined in}} expected-note {{'std::legendrel' is a}}
  std::length_error; // expected-error {{no member}} expected-note {{'std::length_error' is defined in}}
  std::lerp; // expected-error {{no member}} expected-note {{'std::lerp' is defined in}} expected-note {{'std::lerp' is a}}
  std::less; // expected-error {{no member}} expected-note {{'std::less' is defined in}}
  std::less_equal; // expected-error {{no member}} expected-note {{'std::less_equal' is defined in}}
  std::lexicographical_compare; // expected-error {{no member}} expected-note {{'std::lexicographical_compare' is defined in}}
  std::lexicographical_compare_three_way; // expected-error {{no member}} expected-note {{'std::lexicographical_compare_three_way' is defined in}} expected-note {{'std::lexicographical_compare_three_way' is a}}
  std::lgammaf; // expected-error {{no member}} expected-note {{'std::lgammaf' is defined in}} expected-note {{'std::lgammaf' is a}}
  std::lgammal; // expected-error {{no member}} expected-note {{'std::lgammal' is defined in}} expected-note {{'std::lgammal' is a}}
  std::linear_congruential_engine; // expected-error {{no member}} expected-note {{'std::linear_congruential_engine' is defined in}} expected-note {{'std::linear_congruential_engine' is a}}
  std::list; // expected-error {{no member}} expected-note {{'std::list' is defined in}}
  std::llabs; // expected-error {{no member}} expected-note {{'std::llabs' is defined in}} expected-note {{'std::llabs' is a}}
  std::lldiv; // expected-error {{no member}} expected-note {{'std::lldiv' is defined in}} expected-note {{'std::lldiv' is a}}
  std::lldiv_t; // expected-error {{no member}} expected-note {{'std::lldiv_t' is defined in}} expected-note {{'std::lldiv_t' is a}}
  std::llrint; // expected-error {{no member}} expected-note {{'std::llrint' is defined in}} expected-note {{'std::llrint' is a}}
  std::llrintf; // expected-error {{no member}} expected-note {{'std::llrintf' is defined in}} expected-note {{'std::llrintf' is a}}
  std::llrintl; // expected-error {{no member}} expected-note {{'std::llrintl' is defined in}} expected-note {{'std::llrintl' is a}}
  std::llround; // expected-error {{no member}} expected-note {{'std::llround' is defined in}} expected-note {{'std::llround' is a}}
  std::llroundf; // expected-error {{no member}} expected-note {{'std::llroundf' is defined in}} expected-note {{'std::llroundf' is a}}
  std::llroundl; // expected-error {{no member}} expected-note {{'std::llroundl' is defined in}} expected-note {{'std::llroundl' is a}}
  std::locale; // expected-error {{no member}} expected-note {{'std::locale' is defined in}}
  std::localeconv; // expected-error {{no member}} expected-note {{'std::localeconv' is defined in}}
  std::localtime; // expected-error {{no member}} expected-note {{'std::localtime' is defined in}}
  std::lock; // expected-error {{no member}} expected-note {{'std::lock' is defined in}} expected-note {{'std::lock' is a}}
  std::lock_guard; // expected-error {{no member}} expected-note {{'std::lock_guard' is defined in}} expected-note {{'std::lock_guard' is a}}
  std::log10f; // expected-error {{no member}} expected-note {{'std::log10f' is defined in}} expected-note {{'std::log10f' is a}}
  std::log10l; // expected-error {{no member}} expected-note {{'std::log10l' is defined in}} expected-note {{'std::log10l' is a}}
  std::log1pf; // expected-error {{no member}} expected-note {{'std::log1pf' is defined in}} expected-note {{'std::log1pf' is a}}
  std::log1pl; // expected-error {{no member}} expected-note {{'std::log1pl' is defined in}} expected-note {{'std::log1pl' is a}}
  std::log2f; // expected-error {{no member}} expected-note {{'std::log2f' is defined in}} expected-note {{'std::log2f' is a}}
  std::log2l; // expected-error {{no member}} expected-note {{'std::log2l' is defined in}} expected-note {{'std::log2l' is a}}
  std::logbf; // expected-error {{no member}} expected-note {{'std::logbf' is defined in}} expected-note {{'std::logbf' is a}}
  std::logbl; // expected-error {{no member}} expected-note {{'std::logbl' is defined in}} expected-note {{'std::logbl' is a}}
  std::logf; // expected-error {{no member}} expected-note {{'std::logf' is defined in}} expected-note {{'std::logf' is a}}
  std::logic_error; // expected-error {{no member}} expected-note {{'std::logic_error' is defined in}}
  std::logical_and; // expected-error {{no member}} expected-note {{'std::logical_and' is defined in}}
  std::logical_not; // expected-error {{no member}} expected-note {{'std::logical_not' is defined in}}
  std::logical_or; // expected-error {{no member}} expected-note {{'std::logical_or' is defined in}}
  std::logl; // expected-error {{no member}} expected-note {{'std::logl' is defined in}} expected-note {{'std::logl' is a}}
  std::lognormal_distribution; // expected-error {{no member}} expected-note {{'std::lognormal_distribution' is defined in}} expected-note {{'std::lognormal_distribution' is a}}
  std::longjmp; // expected-error {{no member}} expected-note {{'std::longjmp' is defined in}}
  std::lower_bound; // expected-error {{no member}} expected-note {{'std::lower_bound' is defined in}}
  std::lrint; // expected-error {{no member}} expected-note {{'std::lrint' is defined in}} expected-note {{'std::lrint' is a}}
  std::lrintf; // expected-error {{no member}} expected-note {{'std::lrintf' is defined in}} expected-note {{'std::lrintf' is a}}
  std::lrintl; // expected-error {{no member}} expected-note {{'std::lrintl' is defined in}} expected-note {{'std::lrintl' is a}}
  std::lround; // expected-error {{no member}} expected-note {{'std::lround' is defined in}} expected-note {{'std::lround' is a}}
  std::lroundf; // expected-error {{no member}} expected-note {{'std::lroundf' is defined in}} expected-note {{'std::lroundf' is a}}
  std::lroundl; // expected-error {{no member}} expected-note {{'std::lroundl' is defined in}} expected-note {{'std::lroundl' is a}}
  std::make_any; // expected-error {{no member}} expected-note {{'std::make_any' is defined in}} expected-note {{'std::make_any' is a}}
  std::make_const_iterator; // expected-error {{no member}} expected-note {{'std::make_const_iterator' is defined in}} expected-note {{'std::make_const_iterator' is a}}
  std::make_const_sentinel; // expected-error {{no member}} expected-note {{'std::make_const_sentinel' is defined in}} expected-note {{'std::make_const_sentinel' is a}}
  std::make_exception_ptr; // expected-error {{no member}} expected-note {{'std::make_exception_ptr' is defined in}} expected-note {{'std::make_exception_ptr' is a}}
  std::make_format_args; // expected-error {{no member}} expected-note {{'std::make_format_args' is defined in}} expected-note {{'std::make_format_args' is a}}
  std::make_from_tuple; // expected-error {{no member}} expected-note {{'std::make_from_tuple' is defined in}} expected-note {{'std::make_from_tuple' is a}}
  std::make_heap; // expected-error {{no member}} expected-note {{'std::make_heap' is defined in}}
  std::make_index_sequence; // expected-error {{no member}} expected-note {{'std::make_index_sequence' is defined in}} expected-note {{'std::make_index_sequence' is a}}
  std::make_integer_sequence; // expected-error {{no member}} expected-note {{'std::make_integer_sequence' is defined in}} expected-note {{'std::make_integer_sequence' is a}}
  std::make_move_iterator; // expected-error {{no member}} expected-note {{'std::make_move_iterator' is defined in}} expected-note {{'std::make_move_iterator' is a}}
  std::make_obj_using_allocator; // expected-error {{no member}} expected-note {{'std::make_obj_using_allocator' is defined in}} expected-note {{'std::make_obj_using_allocator' is a}}
  std::make_optional; // expected-error {{no member}} expected-note {{'std::make_optional' is defined in}} expected-note {{'std::make_optional' is a}}
  std::make_pair; // expected-error {{no member}} expected-note {{'std::make_pair' is defined in}}
  std::make_reverse_iterator; // expected-error {{no member}} expected-note {{'std::make_reverse_iterator' is defined in}} expected-note {{'std::make_reverse_iterator' is a}}
  std::make_shared; // expected-error {{no member}} expected-note {{'std::make_shared' is defined in}} expected-note {{'std::make_shared' is a}}
  std::make_shared_for_overwrite; // expected-error {{no member}} expected-note {{'std::make_shared_for_overwrite' is defined in}} expected-note {{'std::make_shared_for_overwrite' is a}}
  std::make_signed; // expected-error {{no member}} expected-note {{'std::make_signed' is defined in}} expected-note {{'std::make_signed' is a}}
  std::make_signed_t; // expected-error {{no member}} expected-note {{'std::make_signed_t' is defined in}} expected-note {{'std::make_signed_t' is a}}
  std::make_tuple; // expected-error {{no member}} expected-note {{'std::make_tuple' is defined in}} expected-note {{'std::make_tuple' is a}}
  std::make_unique; // expected-error {{no member}} expected-note {{'std::make_unique' is defined in}} expected-note {{'std::make_unique' is a}}
  std::make_unique_for_overwrite; // expected-error {{no member}} expected-note {{'std::make_unique_for_overwrite' is defined in}} expected-note {{'std::make_unique_for_overwrite' is a}}
  std::make_unsigned; // expected-error {{no member}} expected-note {{'std::make_unsigned' is defined in}} expected-note {{'std::make_unsigned' is a}}
  std::make_unsigned_t; // expected-error {{no member}} expected-note {{'std::make_unsigned_t' is defined in}} expected-note {{'std::make_unsigned_t' is a}}
  std::make_wformat_args; // expected-error {{no member}} expected-note {{'std::make_wformat_args' is defined in}} expected-note {{'std::make_wformat_args' is a}}
  std::malloc; // expected-error {{no member}} expected-note {{'std::malloc' is defined in}}
  std::map; // expected-error {{no member}} expected-note {{'std::map' is defined in}}
  std::mask_array; // expected-error {{no member}} expected-note {{'std::mask_array' is defined in}}
  std::match_results; // expected-error {{no member}} expected-note {{'std::match_results' is defined in}} expected-note {{'std::match_results' is a}}
  std::max; // expected-error {{no member}} expected-note {{'std::max' is defined in}}
  std::max_align_t; // expected-error {{no member}} expected-note {{'std::max_align_t' is defined in}} expected-note {{'std::max_align_t' is a}}
  std::max_element; // expected-error {{no member}} expected-note {{'std::max_element' is defined in}}
  std::mblen; // expected-error {{no member}} expected-note {{'std::mblen' is defined in}}
  std::mbrlen; // expected-error {{no member}} expected-note {{'std::mbrlen' is defined in}}
  std::mbrtoc16; // expected-error {{no member}} expected-note {{'std::mbrtoc16' is defined in}} expected-note {{'std::mbrtoc16' is a}}
  std::mbrtoc32; // expected-error {{no member}} expected-note {{'std::mbrtoc32' is defined in}} expected-note {{'std::mbrtoc32' is a}}
  std::mbrtoc8; // expected-error {{no member}} expected-note {{'std::mbrtoc8' is defined in}} expected-note {{'std::mbrtoc8' is a}}
  std::mbrtowc; // expected-error {{no member}} expected-note {{'std::mbrtowc' is defined in}}
  std::mbsinit; // expected-error {{no member}} expected-note {{'std::mbsinit' is defined in}}
  std::mbsrtowcs; // expected-error {{no member}} expected-note {{'std::mbsrtowcs' is defined in}}
  std::mbstowcs; // expected-error {{no member}} expected-note {{'std::mbstowcs' is defined in}}
  std::mbtowc; // expected-error {{no member}} expected-note {{'std::mbtowc' is defined in}}
  std::mdspan; // expected-error {{no member}} expected-note {{'std::mdspan' is defined in}} expected-note {{'std::mdspan' is a}}
  std::mega; // expected-error {{no member}} expected-note {{'std::mega' is defined in}} expected-note {{'std::mega' is a}}
  std::mem_fn; // expected-error {{no member}} expected-note {{'std::mem_fn' is defined in}} expected-note {{'std::mem_fn' is a}}
  std::mem_fun; // expected-error {{no member}} expected-note {{'std::mem_fun' is defined in}}
  std::mem_fun1_ref_t; // expected-error {{no member}} expected-note {{'std::mem_fun1_ref_t' is defined in}}
  std::mem_fun1_t; // expected-error {{no member}} expected-note {{'std::mem_fun1_t' is defined in}}
  std::mem_fun_ref; // expected-error {{no member}} expected-note {{'std::mem_fun_ref' is defined in}}
  std::mem_fun_ref_t; // expected-error {{no member}} expected-note {{'std::mem_fun_ref_t' is defined in}}
  std::mem_fun_t; // expected-error {{no member}} expected-note {{'std::mem_fun_t' is defined in}}
  std::memchr; // expected-error {{no member}} expected-note {{'std::memchr' is defined in}}
  std::memcmp; // expected-error {{no member}} expected-note {{'std::memcmp' is defined in}}
  std::memcpy; // expected-error {{no member}} expected-note {{'std::memcpy' is defined in}}
  std::memmove; // expected-error {{no member}} expected-note {{'std::memmove' is defined in}}
  std::memory_order; // expected-error {{no member}} expected-note {{'std::memory_order' is defined in}} expected-note {{'std::memory_order' is a}}
  std::memory_order_acq_rel; // expected-error {{no member}} expected-note {{'std::memory_order_acq_rel' is defined in}} expected-note {{'std::memory_order_acq_rel' is a}}
  std::memory_order_acquire; // expected-error {{no member}} expected-note {{'std::memory_order_acquire' is defined in}} expected-note {{'std::memory_order_acquire' is a}}
  std::memory_order_consume; // expected-error {{no member}} expected-note {{'std::memory_order_consume' is defined in}} expected-note {{'std::memory_order_consume' is a}}
  std::memory_order_relaxed; // expected-error {{no member}} expected-note {{'std::memory_order_relaxed' is defined in}} expected-note {{'std::memory_order_relaxed' is a}}
  std::memory_order_release; // expected-error {{no member}} expected-note {{'std::memory_order_release' is defined in}} expected-note {{'std::memory_order_release' is a}}
  std::memory_order_seq_cst; // expected-error {{no member}} expected-note {{'std::memory_order_seq_cst' is defined in}} expected-note {{'std::memory_order_seq_cst' is a}}
  std::memset; // expected-error {{no member}} expected-note {{'std::memset' is defined in}}
  std::merge; // expected-error {{no member}} expected-note {{'std::merge' is defined in}}
  std::mergeable; // expected-error {{no member}} expected-note {{'std::mergeable' is defined in}} expected-note {{'std::mergeable' is a}}
  std::mersenne_twister_engine; // expected-error {{no member}} expected-note {{'std::mersenne_twister_engine' is defined in}} expected-note {{'std::mersenne_twister_engine' is a}}
  std::messages; // expected-error {{no member}} expected-note {{'std::messages' is defined in}}
  std::messages_base; // expected-error {{no member}} expected-note {{'std::messages_base' is defined in}}
  std::messages_byname; // expected-error {{no member}} expected-note {{'std::messages_byname' is defined in}}
  std::micro; // expected-error {{no member}} expected-note {{'std::micro' is defined in}} expected-note {{'std::micro' is a}}
  std::midpoint; // expected-error {{no member}} expected-note {{'std::midpoint' is defined in}} expected-note {{'std::midpoint' is a}}
  std::milli; // expected-error {{no member}} expected-note {{'std::milli' is defined in}} expected-note {{'std::milli' is a}}
  std::min; // expected-error {{no member}} expected-note {{'std::min' is defined in}}
  std::min_element; // expected-error {{no member}} expected-note {{'std::min_element' is defined in}}
  std::minmax; // expected-error {{no member}} expected-note {{'std::minmax' is defined in}} expected-note {{'std::minmax' is a}}
  std::minmax_element; // expected-error {{no member}} expected-note {{'std::minmax_element' is defined in}} expected-note {{'std::minmax_element' is a}}
  std::minstd_rand; // expected-error {{no member}} expected-note {{'std::minstd_rand' is defined in}} expected-note {{'std::minstd_rand' is a}}
  std::minstd_rand0; // expected-error {{no member}} expected-note {{'std::minstd_rand0' is defined in}} expected-note {{'std::minstd_rand0' is a}}
  std::minus; // expected-error {{no member}} expected-note {{'std::minus' is defined in}}
  std::mismatch; // expected-error {{no member}} expected-note {{'std::mismatch' is defined in}}
  std::mktime; // expected-error {{no member}} expected-note {{'std::mktime' is defined in}}
  std::modf; // expected-error {{no member}} expected-note {{'std::modf' is defined in}}
  std::modff; // expected-error {{no member}} expected-note {{'std::modff' is defined in}} expected-note {{'std::modff' is a}}
  std::modfl; // expected-error {{no member}} expected-note {{'std::modfl' is defined in}} expected-note {{'std::modfl' is a}}
  std::modulus; // expected-error {{no member}} expected-note {{'std::modulus' is defined in}}
  std::money_base; // expected-error {{no member}} expected-note {{'std::money_base' is defined in}}
  std::money_get; // expected-error {{no member}} expected-note {{'std::money_get' is defined in}}
  std::money_put; // expected-error {{no member}} expected-note {{'std::money_put' is defined in}}
  std::moneypunct; // expected-error {{no member}} expected-note {{'std::moneypunct' is defined in}}
  std::moneypunct_byname; // expected-error {{no member}} expected-note {{'std::moneypunct_byname' is defined in}}
  std::movable; // expected-error {{no member}} expected-note {{'std::movable' is defined in}} expected-note {{'std::movable' is a}}
  std::move_backward; // expected-error {{no member}} expected-note {{'std::move_backward' is defined in}} expected-note {{'std::move_backward' is a}}
  std::move_constructible; // expected-error {{no member}} expected-note {{'std::move_constructible' is defined in}} expected-note {{'std::move_constructible' is a}}
  std::move_if_noexcept; // expected-error {{no member}} expected-note {{'std::move_if_noexcept' is defined in}} expected-note {{'std::move_if_noexcept' is a}}
  std::move_iterator; // expected-error {{no member}} expected-note {{'std::move_iterator' is defined in}} expected-note {{'std::move_iterator' is a}}
  std::move_only_function; // expected-error {{no member}} expected-note {{'std::move_only_function' is defined in}} expected-note {{'std::move_only_function' is a}}
  std::move_sentinel; // expected-error {{no member}} expected-note {{'std::move_sentinel' is defined in}} expected-note {{'std::move_sentinel' is a}}
  std::mt19937; // expected-error {{no member}} expected-note {{'std::mt19937' is defined in}} expected-note {{'std::mt19937' is a}}
  std::mt19937_64; // expected-error {{no member}} expected-note {{'std::mt19937_64' is defined in}} expected-note {{'std::mt19937_64' is a}}
  std::mul_sat; // expected-error {{no member}} expected-note {{'std::mul_sat' is defined in}} expected-note {{'std::mul_sat' is a}}
  std::multimap; // expected-error {{no member}} expected-note {{'std::multimap' is defined in}}
  std::multiplies; // expected-error {{no member}} expected-note {{'std::multiplies' is defined in}}
  std::multiset; // expected-error {{no member}} expected-note {{'std::multiset' is defined in}}
  std::mutex; // expected-error {{no member}} expected-note {{'std::mutex' is defined in}} expected-note {{'std::mutex' is a}}
  std::nan; // expected-error {{no member}} expected-note {{'std::nan' is defined in}} expected-note {{'std::nan' is a}}
  std::nanf; // expected-error {{no member}} expected-note {{'std::nanf' is defined in}} expected-note {{'std::nanf' is a}}
  std::nanl; // expected-error {{no member}} expected-note {{'std::nanl' is defined in}} expected-note {{'std::nanl' is a}}
  std::nano; // expected-error {{no member}} expected-note {{'std::nano' is defined in}} expected-note {{'std::nano' is a}}
  std::nearbyintf; // expected-error {{no member}} expected-note {{'std::nearbyintf' is defined in}} expected-note {{'std::nearbyintf' is a}}
  std::nearbyintl; // expected-error {{no member}} expected-note {{'std::nearbyintl' is defined in}} expected-note {{'std::nearbyintl' is a}}
  std::negate; // expected-error {{no member}} expected-note {{'std::negate' is defined in}}
  std::negation; // expected-error {{no member}} expected-note {{'std::negation' is defined in}} expected-note {{'std::negation' is a}}
  std::negation_v; // expected-error {{no member}} expected-note {{'std::negation_v' is defined in}} expected-note {{'std::negation_v' is a}}
  std::negative_binomial_distribution; // expected-error {{no member}} expected-note {{'std::negative_binomial_distribution' is defined in}} expected-note {{'std::negative_binomial_distribution' is a}}
  std::nested_exception; // expected-error {{no member}} expected-note {{'std::nested_exception' is defined in}} expected-note {{'std::nested_exception' is a}}
  std::new_handler; // expected-error {{no member}} expected-note {{'std::new_handler' is defined in}}
  std::next; // expected-error {{no member}} expected-note {{'std::next' is defined in}} expected-note {{'std::next' is a}}
  std::next_permutation; // expected-error {{no member}} expected-note {{'std::next_permutation' is defined in}}
  std::nextafter; // expected-error {{no member}} expected-note {{'std::nextafter' is defined in}} expected-note {{'std::nextafter' is a}}
  std::nextafterf; // expected-error {{no member}} expected-note {{'std::nextafterf' is defined in}} expected-note {{'std::nextafterf' is a}}
  std::nextafterl; // expected-error {{no member}} expected-note {{'std::nextafterl' is defined in}} expected-note {{'std::nextafterl' is a}}
  std::nexttoward; // expected-error {{no member}} expected-note {{'std::nexttoward' is defined in}} expected-note {{'std::nexttoward' is a}}
  std::nexttowardf; // expected-error {{no member}} expected-note {{'std::nexttowardf' is defined in}} expected-note {{'std::nexttowardf' is a}}
  std::nexttowardl; // expected-error {{no member}} expected-note {{'std::nexttowardl' is defined in}} expected-note {{'std::nexttowardl' is a}}
  std::noboolalpha; // expected-error {{no member}} expected-note {{'std::noboolalpha' is defined in}}
  std::noboolalpha; // expected-error {{no member}} expected-note {{'std::noboolalpha' is defined in}}
  std::noemit_on_flush; // expected-error {{no member}} expected-note {{'std::noemit_on_flush' is defined in}} expected-note {{'std::noemit_on_flush' is a}}
  std::noemit_on_flush; // expected-error {{no member}} expected-note {{'std::noemit_on_flush' is defined in}} expected-note {{'std::noemit_on_flush' is a}}
  std::none_of; // expected-error {{no member}} expected-note {{'std::none_of' is defined in}} expected-note {{'std::none_of' is a}}
  std::nontype; // expected-error {{no member}} expected-note {{'std::nontype' is defined in}} expected-note {{'std::nontype' is a}}
  std::nontype_t; // expected-error {{no member}} expected-note {{'std::nontype_t' is defined in}} expected-note {{'std::nontype_t' is a}}
  std::noop_coroutine; // expected-error {{no member}} expected-note {{'std::noop_coroutine' is defined in}} expected-note {{'std::noop_coroutine' is a}}
  std::noop_coroutine_handle; // expected-error {{no member}} expected-note {{'std::noop_coroutine_handle' is defined in}} expected-note {{'std::noop_coroutine_handle' is a}}
  std::noop_coroutine_promise; // expected-error {{no member}} expected-note {{'std::noop_coroutine_promise' is defined in}} expected-note {{'std::noop_coroutine_promise' is a}}
  std::norm; // expected-error {{no member}} expected-note {{'std::norm' is defined in}}
  std::normal_distribution; // expected-error {{no member}} expected-note {{'std::normal_distribution' is defined in}} expected-note {{'std::normal_distribution' is a}}
  std::noshowbase; // expected-error {{no member}} expected-note {{'std::noshowbase' is defined in}}
  std::noshowbase; // expected-error {{no member}} expected-note {{'std::noshowbase' is defined in}}
  std::noshowpoint; // expected-error {{no member}} expected-note {{'std::noshowpoint' is defined in}}
  std::noshowpoint; // expected-error {{no member}} expected-note {{'std::noshowpoint' is defined in}}
  std::noshowpos; // expected-error {{no member}} expected-note {{'std::noshowpos' is defined in}}
  std::noshowpos; // expected-error {{no member}} expected-note {{'std::noshowpos' is defined in}}
  std::noskipws; // expected-error {{no member}} expected-note {{'std::noskipws' is defined in}}
  std::noskipws; // expected-error {{no member}} expected-note {{'std::noskipws' is defined in}}
  std::nostopstate; // expected-error {{no member}} expected-note {{'std::nostopstate' is defined in}} expected-note {{'std::nostopstate' is a}}
  std::nostopstate_t; // expected-error {{no member}} expected-note {{'std::nostopstate_t' is defined in}} expected-note {{'std::nostopstate_t' is a}}
  std::not1; // expected-error {{no member}} expected-note {{'std::not1' is defined in}}
  std::not2; // expected-error {{no member}} expected-note {{'std::not2' is defined in}}
  std::not_equal_to; // expected-error {{no member}} expected-note {{'std::not_equal_to' is defined in}}
  std::not_fn; // expected-error {{no member}} expected-note {{'std::not_fn' is defined in}} expected-note {{'std::not_fn' is a}}
  std::nothrow; // expected-error {{no member}} expected-note {{'std::nothrow' is defined in}}
  std::nothrow_t; // expected-error {{no member}} expected-note {{'std::nothrow_t' is defined in}}
  std::notify_all_at_thread_exit; // expected-error {{no member}} expected-note {{'std::notify_all_at_thread_exit' is defined in}} expected-note {{'std::notify_all_at_thread_exit' is a}}
  std::nounitbuf; // expected-error {{no member}} expected-note {{'std::nounitbuf' is defined in}}
  std::nounitbuf; // expected-error {{no member}} expected-note {{'std::nounitbuf' is defined in}}
  std::nouppercase; // expected-error {{no member}} expected-note {{'std::nouppercase' is defined in}}
  std::nouppercase; // expected-error {{no member}} expected-note {{'std::nouppercase' is defined in}}
  std::nth_element; // expected-error {{no member}} expected-note {{'std::nth_element' is defined in}}
  std::nullopt; // expected-error {{no member}} expected-note {{'std::nullopt' is defined in}} expected-note {{'std::nullopt' is a}}
  std::nullopt_t; // expected-error {{no member}} expected-note {{'std::nullopt_t' is defined in}} expected-note {{'std::nullopt_t' is a}}
  std::nullptr_t; // expected-error {{no member}} expected-note {{'std::nullptr_t' is defined in}} expected-note {{'std::nullptr_t' is a}}
  std::num_get; // expected-error {{no member}} expected-note {{'std::num_get' is defined in}}
  std::num_put; // expected-error {{no member}} expected-note {{'std::num_put' is defined in}}
  std::numeric_limits; // expected-error {{no member}} expected-note {{'std::numeric_limits' is defined in}}
  std::numpunct; // expected-error {{no member}} expected-note {{'std::numpunct' is defined in}}
  std::numpunct_byname; // expected-error {{no member}} expected-note {{'std::numpunct_byname' is defined in}}
  std::oct; // expected-error {{no member}} expected-note {{'std::oct' is defined in}}
  std::oct; // expected-error {{no member}} expected-note {{'std::oct' is defined in}}
  std::ofstream; // expected-error {{no member}} expected-note {{'std::ofstream' is defined in}}
  std::ofstream; // expected-error {{no member}} expected-note {{'std::ofstream' is defined in}}
  std::once_flag; // expected-error {{no member}} expected-note {{'std::once_flag' is defined in}} expected-note {{'std::once_flag' is a}}
  std::op; // expected-error {{no member}} expected-note {{'std::op' is defined in}}
  std::open_mode; // expected-error {{no member}} expected-note {{'std::open_mode' is defined in}}
  std::open_mode; // expected-error {{no member}} expected-note {{'std::open_mode' is defined in}}
  std::optional; // expected-error {{no member}} expected-note {{'std::optional' is defined in}} expected-note {{'std::optional' is a}}
  std::ospanstream; // expected-error {{no member}} expected-note {{'std::ospanstream' is defined in}} expected-note {{'std::ospanstream' is a}}
  std::ospanstream; // expected-error {{no member}} expected-note {{'std::ospanstream' is defined in}} expected-note {{'std::ospanstream' is a}}
  std::ostream; // expected-error {{no member}} expected-note {{'std::ostream' is defined in}}
  std::ostream; // expected-error {{no member}} expected-note {{'std::ostream' is defined in}}
  std::ostream; // expected-error {{no member}} expected-note {{'std::ostream' is defined in}}
  std::ostream_iterator; // expected-error {{no member}} expected-note {{'std::ostream_iterator' is defined in}}
  std::ostreambuf_iterator; // expected-error {{no member}} expected-note {{'std::ostreambuf_iterator' is defined in}}
  std::ostreambuf_iterator; // expected-error {{no member}} expected-note {{'std::ostreambuf_iterator' is defined in}}
  std::ostringstream; // expected-error {{no member}} expected-note {{'std::ostringstream' is defined in}}
  std::ostringstream; // expected-error {{no member}} expected-note {{'std::ostringstream' is defined in}}
  std::ostrstream; // expected-error {{no member}} expected-note {{'std::ostrstream' is defined in}}
  std::osyncstream; // expected-error {{no member}} expected-note {{'std::osyncstream' is defined in}} expected-note {{'std::osyncstream' is a}}
  std::osyncstream; // expected-error {{no member}} expected-note {{'std::osyncstream' is defined in}} expected-note {{'std::osyncstream' is a}}
  std::out_of_range; // expected-error {{no member}} expected-note {{'std::out_of_range' is defined in}}
  std::out_ptr; // expected-error {{no member}} expected-note {{'std::out_ptr' is defined in}} expected-note {{'std::out_ptr' is a}}
  std::out_ptr_t; // expected-error {{no member}} expected-note {{'std::out_ptr_t' is defined in}} expected-note {{'std::out_ptr_t' is a}}
  std::output_iterator; // expected-error {{no member}} expected-note {{'std::output_iterator' is defined in}} expected-note {{'std::output_iterator' is a}}
  std::output_iterator_tag; // expected-error {{no member}} expected-note {{'std::output_iterator_tag' is defined in}}
  std::overflow_error; // expected-error {{no member}} expected-note {{'std::overflow_error' is defined in}}
  std::owner_less; // expected-error {{no member}} expected-note {{'std::owner_less' is defined in}} expected-note {{'std::owner_less' is a}}
  std::packaged_task; // expected-error {{no member}} expected-note {{'std::packaged_task' is defined in}} expected-note {{'std::packaged_task' is a}}
  std::pair; // expected-error {{no member}} expected-note {{'std::pair' is defined in}}
  std::partial_order; // expected-error {{no member}} expected-note {{'std::partial_order' is defined in}} expected-note {{'std::partial_order' is a}}
  std::partial_ordering; // expected-error {{no member}} expected-note {{'std::partial_ordering' is defined in}} expected-note {{'std::partial_ordering' is a}}
  std::partial_sort; // expected-error {{no member}} expected-note {{'std::partial_sort' is defined in}}
  std::partial_sort_copy; // expected-error {{no member}} expected-note {{'std::partial_sort_copy' is defined in}}
  std::partial_sum; // expected-error {{no member}} expected-note {{'std::partial_sum' is defined in}}
  std::partition; // expected-error {{no member}} expected-note {{'std::partition' is defined in}}
  std::partition_copy; // expected-error {{no member}} expected-note {{'std::partition_copy' is defined in}} expected-note {{'std::partition_copy' is a}}
  std::partition_point; // expected-error {{no member}} expected-note {{'std::partition_point' is defined in}} expected-note {{'std::partition_point' is a}}
  std::permutable; // expected-error {{no member}} expected-note {{'std::permutable' is defined in}} expected-note {{'std::permutable' is a}}
  std::perror; // expected-error {{no member}} expected-note {{'std::perror' is defined in}}
  std::peta; // expected-error {{no member}} expected-note {{'std::peta' is defined in}} expected-note {{'std::peta' is a}}
  std::pico; // expected-error {{no member}} expected-note {{'std::pico' is defined in}} expected-note {{'std::pico' is a}}
  std::piecewise_constant_distribution; // expected-error {{no member}} expected-note {{'std::piecewise_constant_distribution' is defined in}} expected-note {{'std::piecewise_constant_distribution' is a}}
  std::piecewise_construct; // expected-error {{no member}} expected-note {{'std::piecewise_construct' is defined in}} expected-note {{'std::piecewise_construct' is a}}
  std::piecewise_construct_t; // expected-error {{no member}} expected-note {{'std::piecewise_construct_t' is defined in}} expected-note {{'std::piecewise_construct_t' is a}}
  std::piecewise_linear_distribution; // expected-error {{no member}} expected-note {{'std::piecewise_linear_distribution' is defined in}} expected-note {{'std::piecewise_linear_distribution' is a}}
  std::plus; // expected-error {{no member}} expected-note {{'std::plus' is defined in}}
  std::pointer_safety; // expected-error {{no member}} expected-note {{'std::pointer_safety' is defined in}} expected-note {{'std::pointer_safety' is a}}
  std::pointer_traits; // expected-error {{no member}} expected-note {{'std::pointer_traits' is defined in}} expected-note {{'std::pointer_traits' is a}}
  std::poisson_distribution; // expected-error {{no member}} expected-note {{'std::poisson_distribution' is defined in}} expected-note {{'std::poisson_distribution' is a}}
  std::polar; // expected-error {{no member}} expected-note {{'std::polar' is defined in}}
  std::pop_heap; // expected-error {{no member}} expected-note {{'std::pop_heap' is defined in}}
  std::popcount; // expected-error {{no member}} expected-note {{'std::popcount' is defined in}} expected-note {{'std::popcount' is a}}
  std::pow; // expected-error {{no member}} expected-note {{'std::pow' is defined in}}
  std::powf; // expected-error {{no member}} expected-note {{'std::powf' is defined in}} expected-note {{'std::powf' is a}}
  std::powl; // expected-error {{no member}} expected-note {{'std::powl' is defined in}} expected-note {{'std::powl' is a}}
  std::predicate; // expected-error {{no member}} expected-note {{'std::predicate' is defined in}} expected-note {{'std::predicate' is a}}
  std::preferred; // expected-error {{no member}} expected-note {{'std::preferred' is defined in}} expected-note {{'std::preferred' is a}}
  std::prev; // expected-error {{no member}} expected-note {{'std::prev' is defined in}} expected-note {{'std::prev' is a}}
  std::prev_permutation; // expected-error {{no member}} expected-note {{'std::prev_permutation' is defined in}}
  std::print; // expected-error {{no member}} expected-note {{'std::print' is defined in}} expected-note {{'std::print' is a}}
  std::printf; // expected-error {{no member}} expected-note {{'std::printf' is defined in}}
  std::println; // expected-error {{no member}} expected-note {{'std::println' is defined in}} expected-note {{'std::println' is a}}
  std::priority_queue; // expected-error {{no member}} expected-note {{'std::priority_queue' is defined in}}
  std::proj; // expected-error {{no member}} expected-note {{'std::proj' is defined in}} expected-note {{'std::proj' is a}}
  std::projected; // expected-error {{no member}} expected-note {{'std::projected' is defined in}} expected-note {{'std::projected' is a}}
  std::promise; // expected-error {{no member}} expected-note {{'std::promise' is defined in}} expected-note {{'std::promise' is a}}
  std::ptr_fun; // expected-error {{no member}} expected-note {{'std::ptr_fun' is defined in}}
  std::ptrdiff_t; // expected-error {{no member}} expected-note {{'std::ptrdiff_t' is defined in}}
  std::push_heap; // expected-error {{no member}} expected-note {{'std::push_heap' is defined in}}
  std::put_money; // expected-error {{no member}} expected-note {{'std::put_money' is defined in}} expected-note {{'std::put_money' is a}}
  std::put_time; // expected-error {{no member}} expected-note {{'std::put_time' is defined in}} expected-note {{'std::put_time' is a}}
  std::putc; // expected-error {{no member}} expected-note {{'std::putc' is defined in}}
  std::putchar; // expected-error {{no member}} expected-note {{'std::putchar' is defined in}}
  std::puts; // expected-error {{no member}} expected-note {{'std::puts' is defined in}}
  std::putwc; // expected-error {{no member}} expected-note {{'std::putwc' is defined in}}
  std::putwchar; // expected-error {{no member}} expected-note {{'std::putwchar' is defined in}}
  std::qsort; // expected-error {{no member}} expected-note {{'std::qsort' is defined in}}
  std::quecto; // expected-error {{no member}} expected-note {{'std::quecto' is defined in}} expected-note {{'std::quecto' is a}}
  std::quetta; // expected-error {{no member}} expected-note {{'std::quetta' is defined in}} expected-note {{'std::quetta' is a}}
  std::queue; // expected-error {{no member}} expected-note {{'std::queue' is defined in}}
  std::quick_exit; // expected-error {{no member}} expected-note {{'std::quick_exit' is defined in}} expected-note {{'std::quick_exit' is a}}
  std::quoted; // expected-error {{no member}} expected-note {{'std::quoted' is defined in}} expected-note {{'std::quoted' is a}}
  std::raise; // expected-error {{no member}} expected-note {{'std::raise' is defined in}}
  std::rand; // expected-error {{no member}} expected-note {{'std::rand' is defined in}}
  std::random_access_iterator; // expected-error {{no member}} expected-note {{'std::random_access_iterator' is defined in}} expected-note {{'std::random_access_iterator' is a}}
  std::random_access_iterator_tag; // expected-error {{no member}} expected-note {{'std::random_access_iterator_tag' is defined in}}
  std::random_device; // expected-error {{no member}} expected-note {{'std::random_device' is defined in}} expected-note {{'std::random_device' is a}}
  std::random_shuffle; // expected-error {{no member}} expected-note {{'std::random_shuffle' is defined in}}
  std::range_error; // expected-error {{no member}} expected-note {{'std::range_error' is defined in}}
  std::range_format; // expected-error {{no member}} expected-note {{'std::range_format' is defined in}} expected-note {{'std::range_format' is a}}
  std::range_formatter; // expected-error {{no member}} expected-note {{'std::range_formatter' is defined in}} expected-note {{'std::range_formatter' is a}}
  std::rank; // expected-error {{no member}} expected-note {{'std::rank' is defined in}} expected-note {{'std::rank' is a}}
  std::rank_v; // expected-error {{no member}} expected-note {{'std::rank_v' is defined in}} expected-note {{'std::rank_v' is a}}
  std::ranlux24; // expected-error {{no member}} expected-note {{'std::ranlux24' is defined in}} expected-note {{'std::ranlux24' is a}}
  std::ranlux24_base; // expected-error {{no member}} expected-note {{'std::ranlux24_base' is defined in}} expected-note {{'std::ranlux24_base' is a}}
  std::ranlux48; // expected-error {{no member}} expected-note {{'std::ranlux48' is defined in}} expected-note {{'std::ranlux48' is a}}
  std::ranlux48_base; // expected-error {{no member}} expected-note {{'std::ranlux48_base' is defined in}} expected-note {{'std::ranlux48_base' is a}}
  std::ratio; // expected-error {{no member}} expected-note {{'std::ratio' is defined in}} expected-note {{'std::ratio' is a}}
  std::ratio_add; // expected-error {{no member}} expected-note {{'std::ratio_add' is defined in}} expected-note {{'std::ratio_add' is a}}
  std::ratio_divide; // expected-error {{no member}} expected-note {{'std::ratio_divide' is defined in}} expected-note {{'std::ratio_divide' is a}}
  std::ratio_equal; // expected-error {{no member}} expected-note {{'std::ratio_equal' is defined in}} expected-note {{'std::ratio_equal' is a}}
  std::ratio_equal_v; // expected-error {{no member}} expected-note {{'std::ratio_equal_v' is defined in}} expected-note {{'std::ratio_equal_v' is a}}
  std::ratio_greater; // expected-error {{no member}} expected-note {{'std::ratio_greater' is defined in}} expected-note {{'std::ratio_greater' is a}}
  std::ratio_greater_equal; // expected-error {{no member}} expected-note {{'std::ratio_greater_equal' is defined in}} expected-note {{'std::ratio_greater_equal' is a}}
  std::ratio_greater_equal_v; // expected-error {{no member}} expected-note {{'std::ratio_greater_equal_v' is defined in}} expected-note {{'std::ratio_greater_equal_v' is a}}
  std::ratio_greater_v; // expected-error {{no member}} expected-note {{'std::ratio_greater_v' is defined in}} expected-note {{'std::ratio_greater_v' is a}}
  std::ratio_less; // expected-error {{no member}} expected-note {{'std::ratio_less' is defined in}} expected-note {{'std::ratio_less' is a}}
  std::ratio_less_equal; // expected-error {{no member}} expected-note {{'std::ratio_less_equal' is defined in}} expected-note {{'std::ratio_less_equal' is a}}
  std::ratio_less_equal_v; // expected-error {{no member}} expected-note {{'std::ratio_less_equal_v' is defined in}} expected-note {{'std::ratio_less_equal_v' is a}}
  std::ratio_less_v; // expected-error {{no member}} expected-note {{'std::ratio_less_v' is defined in}} expected-note {{'std::ratio_less_v' is a}}
  std::ratio_multiply; // expected-error {{no member}} expected-note {{'std::ratio_multiply' is defined in}} expected-note {{'std::ratio_multiply' is a}}
  std::ratio_not_equal; // expected-error {{no member}} expected-note {{'std::ratio_not_equal' is defined in}} expected-note {{'std::ratio_not_equal' is a}}
  std::ratio_not_equal_v; // expected-error {{no member}} expected-note {{'std::ratio_not_equal_v' is defined in}} expected-note {{'std::ratio_not_equal_v' is a}}
  std::ratio_subtract; // expected-error {{no member}} expected-note {{'std::ratio_subtract' is defined in}} expected-note {{'std::ratio_subtract' is a}}
  std::raw_storage_iterator; // expected-error {{no member}} expected-note {{'std::raw_storage_iterator' is defined in}}
  std::real; // expected-error {{no member}} expected-note {{'std::real' is defined in}}
  std::realloc; // expected-error {{no member}} expected-note {{'std::realloc' is defined in}}
  std::recursive_mutex; // expected-error {{no member}} expected-note {{'std::recursive_mutex' is defined in}} expected-note {{'std::recursive_mutex' is a}}
  std::recursive_timed_mutex; // expected-error {{no member}} expected-note {{'std::recursive_timed_mutex' is defined in}} expected-note {{'std::recursive_timed_mutex' is a}}
  std::reduce; // expected-error {{no member}} expected-note {{'std::reduce' is defined in}} expected-note {{'std::reduce' is a}}
  std::ref; // expected-error {{no member}} expected-note {{'std::ref' is defined in}} expected-note {{'std::ref' is a}}
  std::reference_constructs_from_temporary; // expected-error {{no member}} expected-note {{'std::reference_constructs_from_temporary' is defined in}} expected-note {{'std::reference_constructs_from_temporary' is a}}
  std::reference_converts_from_temporary; // expected-error {{no member}} expected-note {{'std::reference_converts_from_temporary' is defined in}} expected-note {{'std::reference_converts_from_temporary' is a}}
  std::reference_wrapper; // expected-error {{no member}} expected-note {{'std::reference_wrapper' is defined in}} expected-note {{'std::reference_wrapper' is a}}
  std::regex; // expected-error {{no member}} expected-note {{'std::regex' is defined in}} expected-note {{'std::regex' is a}}
  std::regex_error; // expected-error {{no member}} expected-note {{'std::regex_error' is defined in}} expected-note {{'std::regex_error' is a}}
  std::regex_iterator; // expected-error {{no member}} expected-note {{'std::regex_iterator' is defined in}} expected-note {{'std::regex_iterator' is a}}
  std::regex_match; // expected-error {{no member}} expected-note {{'std::regex_match' is defined in}} expected-note {{'std::regex_match' is a}}
  std::regex_replace; // expected-error {{no member}} expected-note {{'std::regex_replace' is defined in}} expected-note {{'std::regex_replace' is a}}
  std::regex_search; // expected-error {{no member}} expected-note {{'std::regex_search' is defined in}} expected-note {{'std::regex_search' is a}}
  std::regex_token_iterator; // expected-error {{no member}} expected-note {{'std::regex_token_iterator' is defined in}} expected-note {{'std::regex_token_iterator' is a}}
  std::regex_traits; // expected-error {{no member}} expected-note {{'std::regex_traits' is defined in}} expected-note {{'std::regex_traits' is a}}
  std::regular; // expected-error {{no member}} expected-note {{'std::regular' is defined in}} expected-note {{'std::regular' is a}}
  std::regular_invocable; // expected-error {{no member}} expected-note {{'std::regular_invocable' is defined in}} expected-note {{'std::regular_invocable' is a}}
  std::reinterpret_pointer_cast; // expected-error {{no member}} expected-note {{'std::reinterpret_pointer_cast' is defined in}} expected-note {{'std::reinterpret_pointer_cast' is a}}
  std::relation; // expected-error {{no member}} expected-note {{'std::relation' is defined in}} expected-note {{'std::relation' is a}}
  std::relaxed; // expected-error {{no member}} expected-note {{'std::relaxed' is defined in}} expected-note {{'std::relaxed' is a}}
  std::remainderf; // expected-error {{no member}} expected-note {{'std::remainderf' is defined in}} expected-note {{'std::remainderf' is a}}
  std::remainderl; // expected-error {{no member}} expected-note {{'std::remainderl' is defined in}} expected-note {{'std::remainderl' is a}}
  std::remove_all_extents; // expected-error {{no member}} expected-note {{'std::remove_all_extents' is defined in}} expected-note {{'std::remove_all_extents' is a}}
  std::remove_all_extents_t; // expected-error {{no member}} expected-note {{'std::remove_all_extents_t' is defined in}} expected-note {{'std::remove_all_extents_t' is a}}
  std::remove_const; // expected-error {{no member}} expected-note {{'std::remove_const' is defined in}} expected-note {{'std::remove_const' is a}}
  std::remove_const_t; // expected-error {{no member}} expected-note {{'std::remove_const_t' is defined in}} expected-note {{'std::remove_const_t' is a}}
  std::remove_copy; // expected-error {{no member}} expected-note {{'std::remove_copy' is defined in}}
  std::remove_copy_if; // expected-error {{no member}} expected-note {{'std::remove_copy_if' is defined in}}
  std::remove_cv; // expected-error {{no member}} expected-note {{'std::remove_cv' is defined in}} expected-note {{'std::remove_cv' is a}}
  std::remove_cv_t; // expected-error {{no member}} expected-note {{'std::remove_cv_t' is defined in}} expected-note {{'std::remove_cv_t' is a}}
  std::remove_cvref; // expected-error {{no member}} expected-note {{'std::remove_cvref' is defined in}} expected-note {{'std::remove_cvref' is a}}
  std::remove_cvref_t; // expected-error {{no member}} expected-note {{'std::remove_cvref_t' is defined in}} expected-note {{'std::remove_cvref_t' is a}}
  std::remove_extent; // expected-error {{no member}} expected-note {{'std::remove_extent' is defined in}} expected-note {{'std::remove_extent' is a}}
  std::remove_extent_t; // expected-error {{no member}} expected-note {{'std::remove_extent_t' is defined in}} expected-note {{'std::remove_extent_t' is a}}
  std::remove_if; // expected-error {{no member}} expected-note {{'std::remove_if' is defined in}}
  std::remove_pointer; // expected-error {{no member}} expected-note {{'std::remove_pointer' is defined in}} expected-note {{'std::remove_pointer' is a}}
  std::remove_pointer_t; // expected-error {{no member}} expected-note {{'std::remove_pointer_t' is defined in}} expected-note {{'std::remove_pointer_t' is a}}
  std::remove_reference; // expected-error {{no member}} expected-note {{'std::remove_reference' is defined in}} expected-note {{'std::remove_reference' is a}}
  std::remove_reference_t; // expected-error {{no member}} expected-note {{'std::remove_reference_t' is defined in}} expected-note {{'std::remove_reference_t' is a}}
  std::remove_volatile; // expected-error {{no member}} expected-note {{'std::remove_volatile' is defined in}} expected-note {{'std::remove_volatile' is a}}
  std::remove_volatile_t; // expected-error {{no member}} expected-note {{'std::remove_volatile_t' is defined in}} expected-note {{'std::remove_volatile_t' is a}}
  std::remquo; // expected-error {{no member}} expected-note {{'std::remquo' is defined in}} expected-note {{'std::remquo' is a}}
  std::remquof; // expected-error {{no member}} expected-note {{'std::remquof' is defined in}} expected-note {{'std::remquof' is a}}
  std::remquol; // expected-error {{no member}} expected-note {{'std::remquol' is defined in}} expected-note {{'std::remquol' is a}}
  std::rename; // expected-error {{no member}} expected-note {{'std::rename' is defined in}}
  std::replace; // expected-error {{no member}} expected-note {{'std::replace' is defined in}}
  std::replace_copy; // expected-error {{no member}} expected-note {{'std::replace_copy' is defined in}}
  std::replace_copy_if; // expected-error {{no member}} expected-note {{'std::replace_copy_if' is defined in}}
  std::replace_if; // expected-error {{no member}} expected-note {{'std::replace_if' is defined in}}
  std::resetiosflags; // expected-error {{no member}} expected-note {{'std::resetiosflags' is defined in}}
  std::result_of; // expected-error {{no member}} expected-note {{'std::result_of' is defined in}} expected-note {{'std::result_of' is a}}
  std::result_of_t; // expected-error {{no member}} expected-note {{'std::result_of_t' is defined in}} expected-note {{'std::result_of_t' is a}}
  std::rethrow_exception; // expected-error {{no member}} expected-note {{'std::rethrow_exception' is defined in}} expected-note {{'std::rethrow_exception' is a}}
  std::rethrow_if_nested; // expected-error {{no member}} expected-note {{'std::rethrow_if_nested' is defined in}} expected-note {{'std::rethrow_if_nested' is a}}
  std::return_temporary_buffer; // expected-error {{no member}} expected-note {{'std::return_temporary_buffer' is defined in}}
  std::reverse; // expected-error {{no member}} expected-note {{'std::reverse' is defined in}}
  std::reverse_copy; // expected-error {{no member}} expected-note {{'std::reverse_copy' is defined in}}
  std::reverse_iterator; // expected-error {{no member}} expected-note {{'std::reverse_iterator' is defined in}}
  std::rewind; // expected-error {{no member}} expected-note {{'std::rewind' is defined in}}
  std::riemann_zeta; // expected-error {{no member}} expected-note {{'std::riemann_zeta' is defined in}} expected-note {{'std::riemann_zeta' is a}}
  std::riemann_zetaf; // expected-error {{no member}} expected-note {{'std::riemann_zetaf' is defined in}} expected-note {{'std::riemann_zetaf' is a}}
  std::riemann_zetal; // expected-error {{no member}} expected-note {{'std::riemann_zetal' is defined in}} expected-note {{'std::riemann_zetal' is a}}
  std::right; // expected-error {{no member}} expected-note {{'std::right' is defined in}}
  std::right; // expected-error {{no member}} expected-note {{'std::right' is defined in}}
  std::rint; // expected-error {{no member}} expected-note {{'std::rint' is defined in}} expected-note {{'std::rint' is a}}
  std::rintf; // expected-error {{no member}} expected-note {{'std::rintf' is defined in}} expected-note {{'std::rintf' is a}}
  std::rintl; // expected-error {{no member}} expected-note {{'std::rintl' is defined in}} expected-note {{'std::rintl' is a}}
  std::ronna; // expected-error {{no member}} expected-note {{'std::ronna' is defined in}} expected-note {{'std::ronna' is a}}
  std::ronto; // expected-error {{no member}} expected-note {{'std::ronto' is defined in}} expected-note {{'std::ronto' is a}}
  std::rotate; // expected-error {{no member}} expected-note {{'std::rotate' is defined in}}
  std::rotate_copy; // expected-error {{no member}} expected-note {{'std::rotate_copy' is defined in}}
  std::rotl; // expected-error {{no member}} expected-note {{'std::rotl' is defined in}} expected-note {{'std::rotl' is a}}
  std::rotr; // expected-error {{no member}} expected-note {{'std::rotr' is defined in}} expected-note {{'std::rotr' is a}}
  std::round; // expected-error {{no member}} expected-note {{'std::round' is defined in}} expected-note {{'std::round' is a}}
  std::round_indeterminate; // expected-error {{no member}} expected-note {{'std::round_indeterminate' is defined in}}
  std::round_to_nearest; // expected-error {{no member}} expected-note {{'std::round_to_nearest' is defined in}}
  std::round_toward_infinity; // expected-error {{no member}} expected-note {{'std::round_toward_infinity' is defined in}}
  std::round_toward_neg_infinity; // expected-error {{no member}} expected-note {{'std::round_toward_neg_infinity' is defined in}}
  std::round_toward_zero; // expected-error {{no member}} expected-note {{'std::round_toward_zero' is defined in}}
  std::roundf; // expected-error {{no member}} expected-note {{'std::roundf' is defined in}} expected-note {{'std::roundf' is a}}
  std::roundl; // expected-error {{no member}} expected-note {{'std::roundl' is defined in}} expected-note {{'std::roundl' is a}}
  std::runtime_error; // expected-error {{no member}} expected-note {{'std::runtime_error' is defined in}}
  std::runtime_format; // expected-error {{no member}} expected-note {{'std::runtime_format' is defined in}} expected-note {{'std::runtime_format' is a}}
  std::same_as; // expected-error {{no member}} expected-note {{'std::same_as' is defined in}} expected-note {{'std::same_as' is a}}
  std::sample; // expected-error {{no member}} expected-note {{'std::sample' is defined in}} expected-note {{'std::sample' is a}}
  std::saturate_cast; // expected-error {{no member}} expected-note {{'std::saturate_cast' is defined in}} expected-note {{'std::saturate_cast' is a}}
  std::scalbln; // expected-error {{no member}} expected-note {{'std::scalbln' is defined in}} expected-note {{'std::scalbln' is a}}
  std::scalblnf; // expected-error {{no member}} expected-note {{'std::scalblnf' is defined in}} expected-note {{'std::scalblnf' is a}}
  std::scalblnl; // expected-error {{no member}} expected-note {{'std::scalblnl' is defined in}} expected-note {{'std::scalblnl' is a}}
  std::scalbn; // expected-error {{no member}} expected-note {{'std::scalbn' is defined in}} expected-note {{'std::scalbn' is a}}
  std::scalbnf; // expected-error {{no member}} expected-note {{'std::scalbnf' is defined in}} expected-note {{'std::scalbnf' is a}}
  std::scalbnl; // expected-error {{no member}} expected-note {{'std::scalbnl' is defined in}} expected-note {{'std::scalbnl' is a}}
  std::scanf; // expected-error {{no member}} expected-note {{'std::scanf' is defined in}}
  std::scientific; // expected-error {{no member}} expected-note {{'std::scientific' is defined in}}
  std::scientific; // expected-error {{no member}} expected-note {{'std::scientific' is defined in}}
  std::scoped_allocator_adaptor; // expected-error {{no member}} expected-note {{'std::scoped_allocator_adaptor' is defined in}} expected-note {{'std::scoped_allocator_adaptor' is a}}
  std::scoped_lock; // expected-error {{no member}} expected-note {{'std::scoped_lock' is defined in}} expected-note {{'std::scoped_lock' is a}}
  std::search; // expected-error {{no member}} expected-note {{'std::search' is defined in}}
  std::search_n; // expected-error {{no member}} expected-note {{'std::search_n' is defined in}}
  std::seed_seq; // expected-error {{no member}} expected-note {{'std::seed_seq' is defined in}} expected-note {{'std::seed_seq' is a}}
  std::seek_dir; // expected-error {{no member}} expected-note {{'std::seek_dir' is defined in}}
  std::seek_dir; // expected-error {{no member}} expected-note {{'std::seek_dir' is defined in}}
  std::semiregular; // expected-error {{no member}} expected-note {{'std::semiregular' is defined in}} expected-note {{'std::semiregular' is a}}
  std::sentinel_for; // expected-error {{no member}} expected-note {{'std::sentinel_for' is defined in}} expected-note {{'std::sentinel_for' is a}}
  std::set; // expected-error {{no member}} expected-note {{'std::set' is defined in}}
  std::set_difference; // expected-error {{no member}} expected-note {{'std::set_difference' is defined in}}
  std::set_intersection; // expected-error {{no member}} expected-note {{'std::set_intersection' is defined in}}
  std::set_new_handler; // expected-error {{no member}} expected-note {{'std::set_new_handler' is defined in}}
  std::set_symmetric_difference; // expected-error {{no member}} expected-note {{'std::set_symmetric_difference' is defined in}}
  std::set_terminate; // expected-error {{no member}} expected-note {{'std::set_terminate' is defined in}}
  std::set_unexpected; // expected-error {{no member}} expected-note {{'std::set_unexpected' is defined in}}
  std::set_union; // expected-error {{no member}} expected-note {{'std::set_union' is defined in}}
  std::setbase; // expected-error {{no member}} expected-note {{'std::setbase' is defined in}}
  std::setbuf; // expected-error {{no member}} expected-note {{'std::setbuf' is defined in}}
  std::setfill; // expected-error {{no member}} expected-note {{'std::setfill' is defined in}}
  std::setiosflags; // expected-error {{no member}} expected-note {{'std::setiosflags' is defined in}}
  std::setlocale; // expected-error {{no member}} expected-note {{'std::setlocale' is defined in}}
  std::setprecision; // expected-error {{no member}} expected-note {{'std::setprecision' is defined in}}
  std::setvbuf; // expected-error {{no member}} expected-note {{'std::setvbuf' is defined in}}
  std::setw; // expected-error {{no member}} expected-note {{'std::setw' is defined in}}
  std::shared_future; // expected-error {{no member}} expected-note {{'std::shared_future' is defined in}} expected-note {{'std::shared_future' is a}}
  std::shared_lock; // expected-error {{no member}} expected-note {{'std::shared_lock' is defined in}} expected-note {{'std::shared_lock' is a}}
  std::shared_mutex; // expected-error {{no member}} expected-note {{'std::shared_mutex' is defined in}} expected-note {{'std::shared_mutex' is a}}
  std::shared_ptr; // expected-error {{no member}} expected-note {{'std::shared_ptr' is defined in}} expected-note {{'std::shared_ptr' is a}}
  std::shared_timed_mutex; // expected-error {{no member}} expected-note {{'std::shared_timed_mutex' is defined in}} expected-note {{'std::shared_timed_mutex' is a}}
  std::shift_left; // expected-error {{no member}} expected-note {{'std::shift_left' is defined in}} expected-note {{'std::shift_left' is a}}
  std::shift_right; // expected-error {{no member}} expected-note {{'std::shift_right' is defined in}} expected-note {{'std::shift_right' is a}}
  std::showbase; // expected-error {{no member}} expected-note {{'std::showbase' is defined in}}
  std::showbase; // expected-error {{no member}} expected-note {{'std::showbase' is defined in}}
  std::showpoint; // expected-error {{no member}} expected-note {{'std::showpoint' is defined in}}
  std::showpoint; // expected-error {{no member}} expected-note {{'std::showpoint' is defined in}}
  std::showpos; // expected-error {{no member}} expected-note {{'std::showpos' is defined in}}
  std::showpos; // expected-error {{no member}} expected-note {{'std::showpos' is defined in}}
  std::shuffle; // expected-error {{no member}} expected-note {{'std::shuffle' is defined in}} expected-note {{'std::shuffle' is a}}
  std::shuffle_order_engine; // expected-error {{no member}} expected-note {{'std::shuffle_order_engine' is defined in}} expected-note {{'std::shuffle_order_engine' is a}}
  std::sig_atomic_t; // expected-error {{no member}} expected-note {{'std::sig_atomic_t' is defined in}}
  std::signal; // expected-error {{no member}} expected-note {{'std::signal' is defined in}}
  std::signed_integral; // expected-error {{no member}} expected-note {{'std::signed_integral' is defined in}} expected-note {{'std::signed_integral' is a}}
  std::sinf; // expected-error {{no member}} expected-note {{'std::sinf' is defined in}} expected-note {{'std::sinf' is a}}
  std::sinhf; // expected-error {{no member}} expected-note {{'std::sinhf' is defined in}} expected-note {{'std::sinhf' is a}}
  std::sinhl; // expected-error {{no member}} expected-note {{'std::sinhl' is defined in}} expected-note {{'std::sinhl' is a}}
  std::sinl; // expected-error {{no member}} expected-note {{'std::sinl' is defined in}} expected-note {{'std::sinl' is a}}
  std::sized_sentinel_for; // expected-error {{no member}} expected-note {{'std::sized_sentinel_for' is defined in}} expected-note {{'std::sized_sentinel_for' is a}}
  std::skipws; // expected-error {{no member}} expected-note {{'std::skipws' is defined in}}
  std::skipws; // expected-error {{no member}} expected-note {{'std::skipws' is defined in}}
  std::slice; // expected-error {{no member}} expected-note {{'std::slice' is defined in}}
  std::slice_array; // expected-error {{no member}} expected-note {{'std::slice_array' is defined in}}
  std::smatch; // expected-error {{no member}} expected-note {{'std::smatch' is defined in}} expected-note {{'std::smatch' is a}}
  std::snprintf; // expected-error {{no member}} expected-note {{'std::snprintf' is defined in}} expected-note {{'std::snprintf' is a}}
  std::sort; // expected-error {{no member}} expected-note {{'std::sort' is defined in}}
  std::sort_heap; // expected-error {{no member}} expected-note {{'std::sort_heap' is defined in}}
  std::sortable; // expected-error {{no member}} expected-note {{'std::sortable' is defined in}} expected-note {{'std::sortable' is a}}
  std::source_location; // expected-error {{no member}} expected-note {{'std::source_location' is defined in}} expected-note {{'std::source_location' is a}}
  std::span; // expected-error {{no member}} expected-note {{'std::span' is defined in}} expected-note {{'std::span' is a}}
  std::spanbuf; // expected-error {{no member}} expected-note {{'std::spanbuf' is defined in}} expected-note {{'std::spanbuf' is a}}
  std::spanbuf; // expected-error {{no member}} expected-note {{'std::spanbuf' is defined in}} expected-note {{'std::spanbuf' is a}}
  std::spanstream; // expected-error {{no member}} expected-note {{'std::spanstream' is defined in}} expected-note {{'std::spanstream' is a}}
  std::spanstream; // expected-error {{no member}} expected-note {{'std::spanstream' is defined in}} expected-note {{'std::spanstream' is a}}
  std::sph_bessel; // expected-error {{no member}} expected-note {{'std::sph_bessel' is defined in}} expected-note {{'std::sph_bessel' is a}}
  std::sph_besself; // expected-error {{no member}} expected-note {{'std::sph_besself' is defined in}} expected-note {{'std::sph_besself' is a}}
  std::sph_bessell; // expected-error {{no member}} expected-note {{'std::sph_bessell' is defined in}} expected-note {{'std::sph_bessell' is a}}
  std::sph_legendre; // expected-error {{no member}} expected-note {{'std::sph_legendre' is defined in}} expected-note {{'std::sph_legendre' is a}}
  std::sph_legendref; // expected-error {{no member}} expected-note {{'std::sph_legendref' is defined in}} expected-note {{'std::sph_legendref' is a}}
  std::sph_legendrel; // expected-error {{no member}} expected-note {{'std::sph_legendrel' is defined in}} expected-note {{'std::sph_legendrel' is a}}
  std::sph_neumann; // expected-error {{no member}} expected-note {{'std::sph_neumann' is defined in}} expected-note {{'std::sph_neumann' is a}}
  std::sph_neumannf; // expected-error {{no member}} expected-note {{'std::sph_neumannf' is defined in}} expected-note {{'std::sph_neumannf' is a}}
  std::sph_neumannl; // expected-error {{no member}} expected-note {{'std::sph_neumannl' is defined in}} expected-note {{'std::sph_neumannl' is a}}
  std::sprintf; // expected-error {{no member}} expected-note {{'std::sprintf' is defined in}}
  std::sqrtf; // expected-error {{no member}} expected-note {{'std::sqrtf' is defined in}} expected-note {{'std::sqrtf' is a}}
  std::sqrtl; // expected-error {{no member}} expected-note {{'std::sqrtl' is defined in}} expected-note {{'std::sqrtl' is a}}
  std::srand; // expected-error {{no member}} expected-note {{'std::srand' is defined in}}
  std::sregex_iterator; // expected-error {{no member}} expected-note {{'std::sregex_iterator' is defined in}} expected-note {{'std::sregex_iterator' is a}}
  std::sregex_token_iterator; // expected-error {{no member}} expected-note {{'std::sregex_token_iterator' is defined in}} expected-note {{'std::sregex_token_iterator' is a}}
  std::sscanf; // expected-error {{no member}} expected-note {{'std::sscanf' is defined in}}
  std::ssub_match; // expected-error {{no member}} expected-note {{'std::ssub_match' is defined in}} expected-note {{'std::ssub_match' is a}}
  std::stable_partition; // expected-error {{no member}} expected-note {{'std::stable_partition' is defined in}}
  std::stable_sort; // expected-error {{no member}} expected-note {{'std::stable_sort' is defined in}}
  std::stack; // expected-error {{no member}} expected-note {{'std::stack' is defined in}}
  std::stacktrace; // expected-error {{no member}} expected-note {{'std::stacktrace' is defined in}} expected-note {{'std::stacktrace' is a}}
  std::stacktrace_entry; // expected-error {{no member}} expected-note {{'std::stacktrace_entry' is defined in}} expected-note {{'std::stacktrace_entry' is a}}
  std::start_lifetime_as; // expected-error {{no member}} expected-note {{'std::start_lifetime_as' is defined in}} expected-note {{'std::start_lifetime_as' is a}}
  std::static_pointer_cast; // expected-error {{no member}} expected-note {{'std::static_pointer_cast' is defined in}} expected-note {{'std::static_pointer_cast' is a}}
  std::stod; // expected-error {{no member}} expected-note {{'std::stod' is defined in}} expected-note {{'std::stod' is a}}
  std::stof; // expected-error {{no member}} expected-note {{'std::stof' is defined in}} expected-note {{'std::stof' is a}}
  std::stoi; // expected-error {{no member}} expected-note {{'std::stoi' is defined in}} expected-note {{'std::stoi' is a}}
  std::stol; // expected-error {{no member}} expected-note {{'std::stol' is defined in}} expected-note {{'std::stol' is a}}
  std::stold; // expected-error {{no member}} expected-note {{'std::stold' is defined in}} expected-note {{'std::stold' is a}}
  std::stoll; // expected-error {{no member}} expected-note {{'std::stoll' is defined in}} expected-note {{'std::stoll' is a}}
  std::stop_callback; // expected-error {{no member}} expected-note {{'std::stop_callback' is defined in}} expected-note {{'std::stop_callback' is a}}
  std::stop_source; // expected-error {{no member}} expected-note {{'std::stop_source' is defined in}} expected-note {{'std::stop_source' is a}}
  std::stop_token; // expected-error {{no member}} expected-note {{'std::stop_token' is defined in}} expected-note {{'std::stop_token' is a}}
  std::stoul; // expected-error {{no member}} expected-note {{'std::stoul' is defined in}} expected-note {{'std::stoul' is a}}
  std::stoull; // expected-error {{no member}} expected-note {{'std::stoull' is defined in}} expected-note {{'std::stoull' is a}}
  std::strcat; // expected-error {{no member}} expected-note {{'std::strcat' is defined in}}
  std::strchr; // expected-error {{no member}} expected-note {{'std::strchr' is defined in}}
  std::strcmp; // expected-error {{no member}} expected-note {{'std::strcmp' is defined in}}
  std::strcoll; // expected-error {{no member}} expected-note {{'std::strcoll' is defined in}}
  std::strcpy; // expected-error {{no member}} expected-note {{'std::strcpy' is defined in}}
  std::strcspn; // expected-error {{no member}} expected-note {{'std::strcspn' is defined in}}
  std::streambuf; // expected-error {{no member}} expected-note {{'std::streambuf' is defined in}}
  std::streambuf; // expected-error {{no member}} expected-note {{'std::streambuf' is defined in}}
  std::streambuf; // expected-error {{no member}} expected-note {{'std::streambuf' is defined in}}
  std::streamoff; // expected-error {{no member}} expected-note {{'std::streamoff' is defined in}}
  std::streamoff; // expected-error {{no member}} expected-note {{'std::streamoff' is defined in}}
  std::streampos; // expected-error {{no member}} expected-note {{'std::streampos' is defined in}}
  std::streampos; // expected-error {{no member}} expected-note {{'std::streampos' is defined in}}
  std::streamsize; // expected-error {{no member}} expected-note {{'std::streamsize' is defined in}}
  std::streamsize; // expected-error {{no member}} expected-note {{'std::streamsize' is defined in}}
  std::strerror; // expected-error {{no member}} expected-note {{'std::strerror' is defined in}}
  std::strftime; // expected-error {{no member}} expected-note {{'std::strftime' is defined in}}
  std::strict; // expected-error {{no member}} expected-note {{'std::strict' is defined in}} expected-note {{'std::strict' is a}}
  std::strict_weak_order; // expected-error {{no member}} expected-note {{'std::strict_weak_order' is defined in}} expected-note {{'std::strict_weak_order' is a}}
  std::strided_slice; // expected-error {{no member}} expected-note {{'std::strided_slice' is defined in}} expected-note {{'std::strided_slice' is a}}
  std::string; // expected-error {{no member}} expected-note {{'std::string' is defined in}}
  std::string_view; // expected-error {{no member}} expected-note {{'std::string_view' is defined in}} expected-note {{'std::string_view' is a}}
  std::stringbuf; // expected-error {{no member}} expected-note {{'std::stringbuf' is defined in}}
  std::stringbuf; // expected-error {{no member}} expected-note {{'std::stringbuf' is defined in}}
  std::stringstream; // expected-error {{no member}} expected-note {{'std::stringstream' is defined in}}
  std::stringstream; // expected-error {{no member}} expected-note {{'std::stringstream' is defined in}}
  std::strlen; // expected-error {{no member}} expected-note {{'std::strlen' is defined in}}
  std::strncat; // expected-error {{no member}} expected-note {{'std::strncat' is defined in}}
  std::strncmp; // expected-error {{no member}} expected-note {{'std::strncmp' is defined in}}
  std::strncpy; // expected-error {{no member}} expected-note {{'std::strncpy' is defined in}}
  std::strong_order; // expected-error {{no member}} expected-note {{'std::strong_order' is defined in}} expected-note {{'std::strong_order' is a}}
  std::strong_ordering; // expected-error {{no member}} expected-note {{'std::strong_ordering' is defined in}} expected-note {{'std::strong_ordering' is a}}
  std::strpbrk; // expected-error {{no member}} expected-note {{'std::strpbrk' is defined in}}
  std::strrchr; // expected-error {{no member}} expected-note {{'std::strrchr' is defined in}}
  std::strspn; // expected-error {{no member}} expected-note {{'std::strspn' is defined in}}
  std::strstr; // expected-error {{no member}} expected-note {{'std::strstr' is defined in}}
  std::strstream; // expected-error {{no member}} expected-note {{'std::strstream' is defined in}}
  std::strstreambuf; // expected-error {{no member}} expected-note {{'std::strstreambuf' is defined in}}
  std::strtod; // expected-error {{no member}} expected-note {{'std::strtod' is defined in}}
  std::strtof; // expected-error {{no member}} expected-note {{'std::strtof' is defined in}} expected-note {{'std::strtof' is a}}
  std::strtoimax; // expected-error {{no member}} expected-note {{'std::strtoimax' is defined in}} expected-note {{'std::strtoimax' is a}}
  std::strtok; // expected-error {{no member}} expected-note {{'std::strtok' is defined in}}
  std::strtol; // expected-error {{no member}} expected-note {{'std::strtol' is defined in}}
  std::strtold; // expected-error {{no member}} expected-note {{'std::strtold' is defined in}}
  std::strtoll; // expected-error {{no member}} expected-note {{'std::strtoll' is defined in}} expected-note {{'std::strtoll' is a}}
  std::strtoul; // expected-error {{no member}} expected-note {{'std::strtoul' is defined in}}
  std::strtoull; // expected-error {{no member}} expected-note {{'std::strtoull' is defined in}} expected-note {{'std::strtoull' is a}}
  std::strtoumax; // expected-error {{no member}} expected-note {{'std::strtoumax' is defined in}} expected-note {{'std::strtoumax' is a}}
  std::strxfrm; // expected-error {{no member}} expected-note {{'std::strxfrm' is defined in}}
  std::student_t_distribution; // expected-error {{no member}} expected-note {{'std::student_t_distribution' is defined in}} expected-note {{'std::student_t_distribution' is a}}
  std::sub_match; // expected-error {{no member}} expected-note {{'std::sub_match' is defined in}} expected-note {{'std::sub_match' is a}}
  std::sub_sat; // expected-error {{no member}} expected-note {{'std::sub_sat' is defined in}} expected-note {{'std::sub_sat' is a}}
  std::submdspan_mapping_result; // expected-error {{no member}} expected-note {{'std::submdspan_mapping_result' is defined in}} expected-note {{'std::submdspan_mapping_result' is a}}
  std::subtract_with_carry_engine; // expected-error {{no member}} expected-note {{'std::subtract_with_carry_engine' is defined in}} expected-note {{'std::subtract_with_carry_engine' is a}}
  std::suspend_always; // expected-error {{no member}} expected-note {{'std::suspend_always' is defined in}} expected-note {{'std::suspend_always' is a}}
  std::suspend_never; // expected-error {{no member}} expected-note {{'std::suspend_never' is defined in}} expected-note {{'std::suspend_never' is a}}
  std::swap_ranges; // expected-error {{no member}} expected-note {{'std::swap_ranges' is defined in}}
  std::swappable; // expected-error {{no member}} expected-note {{'std::swappable' is defined in}} expected-note {{'std::swappable' is a}}
  std::swappable_with; // expected-error {{no member}} expected-note {{'std::swappable_with' is defined in}} expected-note {{'std::swappable_with' is a}}
  std::swprintf; // expected-error {{no member}} expected-note {{'std::swprintf' is defined in}}
  std::swscanf; // expected-error {{no member}} expected-note {{'std::swscanf' is defined in}}
  std::syncbuf; // expected-error {{no member}} expected-note {{'std::syncbuf' is defined in}} expected-note {{'std::syncbuf' is a}}
  std::syncbuf; // expected-error {{no member}} expected-note {{'std::syncbuf' is defined in}} expected-note {{'std::syncbuf' is a}}
  std::system; // expected-error {{no member}} expected-note {{'std::system' is defined in}}
  std::system_category; // expected-error {{no member}} expected-note {{'std::system_category' is defined in}} expected-note {{'std::system_category' is a}}
  std::system_error; // expected-error {{no member}} expected-note {{'std::system_error' is defined in}} expected-note {{'std::system_error' is a}}
  std::tanf; // expected-error {{no member}} expected-note {{'std::tanf' is defined in}} expected-note {{'std::tanf' is a}}
  std::tanhf; // expected-error {{no member}} expected-note {{'std::tanhf' is defined in}} expected-note {{'std::tanhf' is a}}
  std::tanhl; // expected-error {{no member}} expected-note {{'std::tanhl' is defined in}} expected-note {{'std::tanhl' is a}}
  std::tanl; // expected-error {{no member}} expected-note {{'std::tanl' is defined in}} expected-note {{'std::tanl' is a}}
  std::tera; // expected-error {{no member}} expected-note {{'std::tera' is defined in}} expected-note {{'std::tera' is a}}
  std::terminate; // expected-error {{no member}} expected-note {{'std::terminate' is defined in}}
  std::terminate_handler; // expected-error {{no member}} expected-note {{'std::terminate_handler' is defined in}}
  std::text_encoding; // expected-error {{no member}} expected-note {{'std::text_encoding' is defined in}} expected-note {{'std::text_encoding' is a}}
  std::tgammaf; // expected-error {{no member}} expected-note {{'std::tgammaf' is defined in}} expected-note {{'std::tgammaf' is a}}
  std::tgammal; // expected-error {{no member}} expected-note {{'std::tgammal' is defined in}} expected-note {{'std::tgammal' is a}}
  std::thread; // expected-error {{no member}} expected-note {{'std::thread' is defined in}} expected-note {{'std::thread' is a}}
  std::three_way_comparable; // expected-error {{no member}} expected-note {{'std::three_way_comparable' is defined in}} expected-note {{'std::three_way_comparable' is a}}
  std::three_way_comparable_with; // expected-error {{no member}} expected-note {{'std::three_way_comparable_with' is defined in}} expected-note {{'std::three_way_comparable_with' is a}}
  std::throw_with_nested; // expected-error {{no member}} expected-note {{'std::throw_with_nested' is defined in}} expected-note {{'std::throw_with_nested' is a}}
  std::tie; // expected-error {{no member}} expected-note {{'std::tie' is defined in}} expected-note {{'std::tie' is a}}
  std::time; // expected-error {{no member}} expected-note {{'std::time' is defined in}}
  std::time_base; // expected-error {{no member}} expected-note {{'std::time_base' is defined in}}
  std::time_get; // expected-error {{no member}} expected-note {{'std::time_get' is defined in}}
  std::time_get_byname; // expected-error {{no member}} expected-note {{'std::time_get_byname' is defined in}}
  std::time_put; // expected-error {{no member}} expected-note {{'std::time_put' is defined in}}
  std::time_put_byname; // expected-error {{no member}} expected-note {{'std::time_put_byname' is defined in}}
  std::time_t; // expected-error {{no member}} expected-note {{'std::time_t' is defined in}}
  std::timed_mutex; // expected-error {{no member}} expected-note {{'std::timed_mutex' is defined in}} expected-note {{'std::timed_mutex' is a}}
  std::timespec; // expected-error {{no member}} expected-note {{'std::timespec' is defined in}} expected-note {{'std::timespec' is a}}
  std::timespec_get; // expected-error {{no member}} expected-note {{'std::timespec_get' is defined in}} expected-note {{'std::timespec_get' is a}}
  std::tm; // expected-error {{no member}} expected-note {{'std::tm' is defined in}}
  std::tmpfile; // expected-error {{no member}} expected-note {{'std::tmpfile' is defined in}}
  std::tmpnam; // expected-error {{no member}} expected-note {{'std::tmpnam' is defined in}}
  std::to_address; // expected-error {{no member}} expected-note {{'std::to_address' is defined in}} expected-note {{'std::to_address' is a}}
  std::to_array; // expected-error {{no member}} expected-note {{'std::to_array' is defined in}} expected-note {{'std::to_array' is a}}
  std::to_chars; // expected-error {{no member}} expected-note {{'std::to_chars' is defined in}} expected-note {{'std::to_chars' is a}}
  std::to_chars_result; // expected-error {{no member}} expected-note {{'std::to_chars_result' is defined in}} expected-note {{'std::to_chars_result' is a}}
  std::to_integer; // expected-error {{no member}} expected-note {{'std::to_integer' is defined in}} expected-note {{'std::to_integer' is a}}
  std::to_string; // expected-error {{no member}} expected-note {{'std::to_string' is defined in}} expected-note {{'std::to_string' is a}}
  std::to_underlying; // expected-error {{no member}} expected-note {{'std::to_underlying' is defined in}} expected-note {{'std::to_underlying' is a}}
  std::to_wstring; // expected-error {{no member}} expected-note {{'std::to_wstring' is defined in}} expected-note {{'std::to_wstring' is a}}
  std::tolower; // expected-error {{no member}} expected-note {{'std::tolower' is defined in}}
  std::totally_ordered; // expected-error {{no member}} expected-note {{'std::totally_ordered' is defined in}} expected-note {{'std::totally_ordered' is a}}
  std::totally_ordered_with; // expected-error {{no member}} expected-note {{'std::totally_ordered_with' is defined in}} expected-note {{'std::totally_ordered_with' is a}}
  std::toupper; // expected-error {{no member}} expected-note {{'std::toupper' is defined in}}
  std::towctrans; // expected-error {{no member}} expected-note {{'std::towctrans' is defined in}}
  std::towlower; // expected-error {{no member}} expected-note {{'std::towlower' is defined in}}
  std::towupper; // expected-error {{no member}} expected-note {{'std::towupper' is defined in}}
  std::transform; // expected-error {{no member}} expected-note {{'std::transform' is defined in}}
  std::transform_exclusive_scan; // expected-error {{no member}} expected-note {{'std::transform_exclusive_scan' is defined in}} expected-note {{'std::transform_exclusive_scan' is a}}
  std::transform_inclusive_scan; // expected-error {{no member}} expected-note {{'std::transform_inclusive_scan' is defined in}} expected-note {{'std::transform_inclusive_scan' is a}}
  std::transform_reduce; // expected-error {{no member}} expected-note {{'std::transform_reduce' is defined in}} expected-note {{'std::transform_reduce' is a}}
  std::true_type; // expected-error {{no member}} expected-note {{'std::true_type' is defined in}} expected-note {{'std::true_type' is a}}
  std::truncf; // expected-error {{no member}} expected-note {{'std::truncf' is defined in}} expected-note {{'std::truncf' is a}}
  std::truncl; // expected-error {{no member}} expected-note {{'std::truncl' is defined in}} expected-note {{'std::truncl' is a}}
  std::try_lock; // expected-error {{no member}} expected-note {{'std::try_lock' is defined in}} expected-note {{'std::try_lock' is a}}
  std::try_to_lock; // expected-error {{no member}} expected-note {{'std::try_to_lock' is defined in}} expected-note {{'std::try_to_lock' is a}}
  std::try_to_lock_t; // expected-error {{no member}} expected-note {{'std::try_to_lock_t' is defined in}} expected-note {{'std::try_to_lock_t' is a}}
  std::tuple; // expected-error {{no member}} expected-note {{'std::tuple' is defined in}} expected-note {{'std::tuple' is a}}
  std::tuple_cat; // expected-error {{no member}} expected-note {{'std::tuple_cat' is defined in}} expected-note {{'std::tuple_cat' is a}}
  std::tuple_element_t; // expected-error {{no member}} expected-note {{'std::tuple_element_t' is defined in}} expected-note {{'std::tuple_element_t' is a}}
  std::tuple_size_v; // expected-error {{no member}} expected-note {{'std::tuple_size_v' is defined in}} expected-note {{'std::tuple_size_v' is a}}
  std::type_identity; // expected-error {{no member}} expected-note {{'std::type_identity' is defined in}} expected-note {{'std::type_identity' is a}}
  std::type_identity_t; // expected-error {{no member}} expected-note {{'std::type_identity_t' is defined in}} expected-note {{'std::type_identity_t' is a}}
  std::type_index; // expected-error {{no member}} expected-note {{'std::type_index' is defined in}} expected-note {{'std::type_index' is a}}
  std::type_info; // expected-error {{no member}} expected-note {{'std::type_info' is defined in}}
  std::u16streampos; // expected-error {{no member}} expected-note {{'std::u16streampos' is defined in}} expected-note {{'std::u16streampos' is a}}
  std::u16streampos; // expected-error {{no member}} expected-note {{'std::u16streampos' is defined in}} expected-note {{'std::u16streampos' is a}}
  std::u16string; // expected-error {{no member}} expected-note {{'std::u16string' is defined in}} expected-note {{'std::u16string' is a}}
  std::u16string_view; // expected-error {{no member}} expected-note {{'std::u16string_view' is defined in}} expected-note {{'std::u16string_view' is a}}
  std::u32streampos; // expected-error {{no member}} expected-note {{'std::u32streampos' is defined in}} expected-note {{'std::u32streampos' is a}}
  std::u32streampos; // expected-error {{no member}} expected-note {{'std::u32streampos' is defined in}} expected-note {{'std::u32streampos' is a}}
  std::u32string; // expected-error {{no member}} expected-note {{'std::u32string' is defined in}} expected-note {{'std::u32string' is a}}
  std::u32string_view; // expected-error {{no member}} expected-note {{'std::u32string_view' is defined in}} expected-note {{'std::u32string_view' is a}}
  std::u8streampos; // expected-error {{no member}} expected-note {{'std::u8streampos' is defined in}} expected-note {{'std::u8streampos' is a}}
  std::u8streampos; // expected-error {{no member}} expected-note {{'std::u8streampos' is defined in}} expected-note {{'std::u8streampos' is a}}
  std::u8string; // expected-error {{no member}} expected-note {{'std::u8string' is defined in}} expected-note {{'std::u8string' is a}}
  std::u8string_view; // expected-error {{no member}} expected-note {{'std::u8string_view' is defined in}} expected-note {{'std::u8string_view' is a}}
  std::uint16_t; // expected-error {{no member}} expected-note {{'std::uint16_t' is defined in}} expected-note {{'std::uint16_t' is a}}
  std::uint32_t; // expected-error {{no member}} expected-note {{'std::uint32_t' is defined in}} expected-note {{'std::uint32_t' is a}}
  std::uint64_t; // expected-error {{no member}} expected-note {{'std::uint64_t' is defined in}} expected-note {{'std::uint64_t' is a}}
  std::uint8_t; // expected-error {{no member}} expected-note {{'std::uint8_t' is defined in}} expected-note {{'std::uint8_t' is a}}
  std::uint_fast16_t; // expected-error {{no member}} expected-note {{'std::uint_fast16_t' is defined in}} expected-note {{'std::uint_fast16_t' is a}}
  std::uint_fast32_t; // expected-error {{no member}} expected-note {{'std::uint_fast32_t' is defined in}} expected-note {{'std::uint_fast32_t' is a}}
  std::uint_fast64_t; // expected-error {{no member}} expected-note {{'std::uint_fast64_t' is defined in}} expected-note {{'std::uint_fast64_t' is a}}
  std::uint_fast8_t; // expected-error {{no member}} expected-note {{'std::uint_fast8_t' is defined in}} expected-note {{'std::uint_fast8_t' is a}}
  std::uint_least16_t; // expected-error {{no member}} expected-note {{'std::uint_least16_t' is defined in}} expected-note {{'std::uint_least16_t' is a}}
  std::uint_least32_t; // expected-error {{no member}} expected-note {{'std::uint_least32_t' is defined in}} expected-note {{'std::uint_least32_t' is a}}
  std::uint_least64_t; // expected-error {{no member}} expected-note {{'std::uint_least64_t' is defined in}} expected-note {{'std::uint_least64_t' is a}}
  std::uint_least8_t; // expected-error {{no member}} expected-note {{'std::uint_least8_t' is defined in}} expected-note {{'std::uint_least8_t' is a}}
  std::uintmax_t; // expected-error {{no member}} expected-note {{'std::uintmax_t' is defined in}} expected-note {{'std::uintmax_t' is a}}
  std::uintptr_t; // expected-error {{no member}} expected-note {{'std::uintptr_t' is defined in}} expected-note {{'std::uintptr_t' is a}}
  std::unary_function; // expected-error {{no member}} expected-note {{'std::unary_function' is defined in}}
  std::unary_negate; // expected-error {{no member}} expected-note {{'std::unary_negate' is defined in}}
  std::uncaught_exception; // expected-error {{no member}} expected-note {{'std::uncaught_exception' is defined in}}
  std::uncaught_exceptions; // expected-error {{no member}} expected-note {{'std::uncaught_exceptions' is defined in}} expected-note {{'std::uncaught_exceptions' is a}}
  std::undeclare_no_pointers; // expected-error {{no member}} expected-note {{'std::undeclare_no_pointers' is defined in}} expected-note {{'std::undeclare_no_pointers' is a}}
  std::undeclare_reachable; // expected-error {{no member}} expected-note {{'std::undeclare_reachable' is defined in}} expected-note {{'std::undeclare_reachable' is a}}
  std::underflow_error; // expected-error {{no member}} expected-note {{'std::underflow_error' is defined in}}
  std::underlying_type; // expected-error {{no member}} expected-note {{'std::underlying_type' is defined in}} expected-note {{'std::underlying_type' is a}}
  std::underlying_type_t; // expected-error {{no member}} expected-note {{'std::underlying_type_t' is defined in}} expected-note {{'std::underlying_type_t' is a}}
  std::unexpect; // expected-error {{no member}} expected-note {{'std::unexpect' is defined in}} expected-note {{'std::unexpect' is a}}
  std::unexpect_t; // expected-error {{no member}} expected-note {{'std::unexpect_t' is defined in}} expected-note {{'std::unexpect_t' is a}}
  std::unexpected; // expected-error {{no member}} expected-note {{'std::unexpected' is defined in}} expected-note {{'std::unexpected' is a}}
  std::unexpected_handler; // expected-error {{no member}} expected-note {{'std::unexpected_handler' is defined in}}
  std::ungetc; // expected-error {{no member}} expected-note {{'std::ungetc' is defined in}}
  std::ungetwc; // expected-error {{no member}} expected-note {{'std::ungetwc' is defined in}}
  std::uniform_int_distribution; // expected-error {{no member}} expected-note {{'std::uniform_int_distribution' is defined in}} expected-note {{'std::uniform_int_distribution' is a}}
  std::uniform_random_bit_generator; // expected-error {{no member}} expected-note {{'std::uniform_random_bit_generator' is defined in}} expected-note {{'std::uniform_random_bit_generator' is a}}
  std::uniform_real_distribution; // expected-error {{no member}} expected-note {{'std::uniform_real_distribution' is defined in}} expected-note {{'std::uniform_real_distribution' is a}}
  std::uninitialized_construct_using_allocator; // expected-error {{no member}} expected-note {{'std::uninitialized_construct_using_allocator' is defined in}} expected-note {{'std::uninitialized_construct_using_allocator' is a}}
  std::uninitialized_copy; // expected-error {{no member}} expected-note {{'std::uninitialized_copy' is defined in}}
  std::uninitialized_copy_n; // expected-error {{no member}} expected-note {{'std::uninitialized_copy_n' is defined in}} expected-note {{'std::uninitialized_copy_n' is a}}
  std::uninitialized_default_construct; // expected-error {{no member}} expected-note {{'std::uninitialized_default_construct' is defined in}} expected-note {{'std::uninitialized_default_construct' is a}}
  std::uninitialized_default_construct_n; // expected-error {{no member}} expected-note {{'std::uninitialized_default_construct_n' is defined in}} expected-note {{'std::uninitialized_default_construct_n' is a}}
  std::uninitialized_fill; // expected-error {{no member}} expected-note {{'std::uninitialized_fill' is defined in}}
  std::uninitialized_fill_n; // expected-error {{no member}} expected-note {{'std::uninitialized_fill_n' is defined in}}
  std::uninitialized_move; // expected-error {{no member}} expected-note {{'std::uninitialized_move' is defined in}} expected-note {{'std::uninitialized_move' is a}}
  std::uninitialized_move_n; // expected-error {{no member}} expected-note {{'std::uninitialized_move_n' is defined in}} expected-note {{'std::uninitialized_move_n' is a}}
  std::uninitialized_value_construct; // expected-error {{no member}} expected-note {{'std::uninitialized_value_construct' is defined in}} expected-note {{'std::uninitialized_value_construct' is a}}
  std::uninitialized_value_construct_n; // expected-error {{no member}} expected-note {{'std::uninitialized_value_construct_n' is defined in}} expected-note {{'std::uninitialized_value_construct_n' is a}}
  std::unique; // expected-error {{no member}} expected-note {{'std::unique' is defined in}}
  std::unique_copy; // expected-error {{no member}} expected-note {{'std::unique_copy' is defined in}}
  std::unique_lock; // expected-error {{no member}} expected-note {{'std::unique_lock' is defined in}} expected-note {{'std::unique_lock' is a}}
  std::unique_ptr; // expected-error {{no member}} expected-note {{'std::unique_ptr' is defined in}} expected-note {{'std::unique_ptr' is a}}
  std::unitbuf; // expected-error {{no member}} expected-note {{'std::unitbuf' is defined in}}
  std::unitbuf; // expected-error {{no member}} expected-note {{'std::unitbuf' is defined in}}
  std::unordered_map; // expected-error {{no member}} expected-note {{'std::unordered_map' is defined in}} expected-note {{'std::unordered_map' is a}}
  std::unordered_multimap; // expected-error {{no member}} expected-note {{'std::unordered_multimap' is defined in}} expected-note {{'std::unordered_multimap' is a}}
  std::unordered_multiset; // expected-error {{no member}} expected-note {{'std::unordered_multiset' is defined in}} expected-note {{'std::unordered_multiset' is a}}
  std::unordered_set; // expected-error {{no member}} expected-note {{'std::unordered_set' is defined in}} expected-note {{'std::unordered_set' is a}}
  std::unreachable; // expected-error {{no member}} expected-note {{'std::unreachable' is defined in}} expected-note {{'std::unreachable' is a}}
  std::unreachable_sentinel; // expected-error {{no member}} expected-note {{'std::unreachable_sentinel' is defined in}} expected-note {{'std::unreachable_sentinel' is a}}
  std::unreachable_sentinel_t; // expected-error {{no member}} expected-note {{'std::unreachable_sentinel_t' is defined in}} expected-note {{'std::unreachable_sentinel_t' is a}}
  std::unsigned_integral; // expected-error {{no member}} expected-note {{'std::unsigned_integral' is defined in}} expected-note {{'std::unsigned_integral' is a}}
  std::upper_bound; // expected-error {{no member}} expected-note {{'std::upper_bound' is defined in}}
  std::uppercase; // expected-error {{no member}} expected-note {{'std::uppercase' is defined in}}
  std::uppercase; // expected-error {{no member}} expected-note {{'std::uppercase' is defined in}}
  std::use_facet; // expected-error {{no member}} expected-note {{'std::use_facet' is defined in}}
  std::uses_allocator; // expected-error {{no member}} expected-note {{'std::uses_allocator' is defined in}} expected-note {{'std::uses_allocator' is a}}
  std::uses_allocator_construction_args; // expected-error {{no member}} expected-note {{'std::uses_allocator_construction_args' is defined in}} expected-note {{'std::uses_allocator_construction_args' is a}}
  std::uses_allocator_v; // expected-error {{no member}} expected-note {{'std::uses_allocator_v' is defined in}} expected-note {{'std::uses_allocator_v' is a}}
  std::va_list; // expected-error {{no member}} expected-note {{'std::va_list' is defined in}}
  std::valarray; // expected-error {{no member}} expected-note {{'std::valarray' is defined in}}
  std::variant; // expected-error {{no member}} expected-note {{'std::variant' is defined in}} expected-note {{'std::variant' is a}}
  std::variant_alternative; // expected-error {{no member}} expected-note {{'std::variant_alternative' is defined in}} expected-note {{'std::variant_alternative' is a}}
  std::variant_alternative_t; // expected-error {{no member}} expected-note {{'std::variant_alternative_t' is defined in}} expected-note {{'std::variant_alternative_t' is a}}
  std::variant_npos; // expected-error {{no member}} expected-note {{'std::variant_npos' is defined in}} expected-note {{'std::variant_npos' is a}}
  std::variant_size; // expected-error {{no member}} expected-note {{'std::variant_size' is defined in}} expected-note {{'std::variant_size' is a}}
  std::variant_size_v; // expected-error {{no member}} expected-note {{'std::variant_size_v' is defined in}} expected-note {{'std::variant_size_v' is a}}
  std::vector; // expected-error {{no member}} expected-note {{'std::vector' is defined in}}
  std::vformat; // expected-error {{no member}} expected-note {{'std::vformat' is defined in}} expected-note {{'std::vformat' is a}}
  std::vformat_to; // expected-error {{no member}} expected-note {{'std::vformat_to' is defined in}} expected-note {{'std::vformat_to' is a}}
  std::vfprintf; // expected-error {{no member}} expected-note {{'std::vfprintf' is defined in}}
  std::vfscanf; // expected-error {{no member}} expected-note {{'std::vfscanf' is defined in}} expected-note {{'std::vfscanf' is a}}
  std::vfwprintf; // expected-error {{no member}} expected-note {{'std::vfwprintf' is defined in}}
  std::vfwscanf; // expected-error {{no member}} expected-note {{'std::vfwscanf' is defined in}} expected-note {{'std::vfwscanf' is a}}
  std::visit; // expected-error {{no member}} expected-note {{'std::visit' is defined in}} expected-note {{'std::visit' is a}}
  std::visit_format_arg; // expected-error {{no member}} expected-note {{'std::visit_format_arg' is defined in}} expected-note {{'std::visit_format_arg' is a}}
  std::void_t; // expected-error {{no member}} expected-note {{'std::void_t' is defined in}} expected-note {{'std::void_t' is a}}
  std::vprint_nonunicode; // expected-error {{no member}} expected-note {{'std::vprint_nonunicode' is defined in}} expected-note {{'std::vprint_nonunicode' is a}}
  std::vprint_nonunicode_buffered; // expected-error {{no member}} expected-note {{'std::vprint_nonunicode_buffered' is defined in}} expected-note {{'std::vprint_nonunicode_buffered' is a}}
  std::vprint_unicode; // expected-error {{no member}} expected-note {{'std::vprint_unicode' is defined in}} expected-note {{'std::vprint_unicode' is a}}
  std::vprint_unicode_buffered; // expected-error {{no member}} expected-note {{'std::vprint_unicode_buffered' is defined in}} expected-note {{'std::vprint_unicode_buffered' is a}}
  std::vprintf; // expected-error {{no member}} expected-note {{'std::vprintf' is defined in}}
  std::vscanf; // expected-error {{no member}} expected-note {{'std::vscanf' is defined in}} expected-note {{'std::vscanf' is a}}
  std::vsnprintf; // expected-error {{no member}} expected-note {{'std::vsnprintf' is defined in}} expected-note {{'std::vsnprintf' is a}}
  std::vsprintf; // expected-error {{no member}} expected-note {{'std::vsprintf' is defined in}}
  std::vsscanf; // expected-error {{no member}} expected-note {{'std::vsscanf' is defined in}} expected-note {{'std::vsscanf' is a}}
  std::vswprintf; // expected-error {{no member}} expected-note {{'std::vswprintf' is defined in}}
  std::vswscanf; // expected-error {{no member}} expected-note {{'std::vswscanf' is defined in}} expected-note {{'std::vswscanf' is a}}
  std::vwprintf; // expected-error {{no member}} expected-note {{'std::vwprintf' is defined in}}
  std::vwscanf; // expected-error {{no member}} expected-note {{'std::vwscanf' is defined in}} expected-note {{'std::vwscanf' is a}}
  std::wbuffer_convert; // expected-error {{no member}} expected-note {{'std::wbuffer_convert' is defined in}}
  std::wbuffer_convert; // expected-error {{no member}} expected-note {{'std::wbuffer_convert' is defined in}}
  std::wcerr; // expected-error {{no member}} expected-note {{'std::wcerr' is defined in}}
  std::wcin; // expected-error {{no member}} expected-note {{'std::wcin' is defined in}}
  std::wclog; // expected-error {{no member}} expected-note {{'std::wclog' is defined in}}
  std::wcmatch; // expected-error {{no member}} expected-note {{'std::wcmatch' is defined in}} expected-note {{'std::wcmatch' is a}}
  std::wcout; // expected-error {{no member}} expected-note {{'std::wcout' is defined in}}
  std::wcregex_iterator; // expected-error {{no member}} expected-note {{'std::wcregex_iterator' is defined in}} expected-note {{'std::wcregex_iterator' is a}}
  std::wcregex_token_iterator; // expected-error {{no member}} expected-note {{'std::wcregex_token_iterator' is defined in}} expected-note {{'std::wcregex_token_iterator' is a}}
  std::wcrtomb; // expected-error {{no member}} expected-note {{'std::wcrtomb' is defined in}}
  std::wcscat; // expected-error {{no member}} expected-note {{'std::wcscat' is defined in}}
  std::wcschr; // expected-error {{no member}} expected-note {{'std::wcschr' is defined in}}
  std::wcscmp; // expected-error {{no member}} expected-note {{'std::wcscmp' is defined in}}
  std::wcscoll; // expected-error {{no member}} expected-note {{'std::wcscoll' is defined in}}
  std::wcscpy; // expected-error {{no member}} expected-note {{'std::wcscpy' is defined in}}
  std::wcscspn; // expected-error {{no member}} expected-note {{'std::wcscspn' is defined in}}
  std::wcsftime; // expected-error {{no member}} expected-note {{'std::wcsftime' is defined in}}
  std::wcslen; // expected-error {{no member}} expected-note {{'std::wcslen' is defined in}}
  std::wcsncat; // expected-error {{no member}} expected-note {{'std::wcsncat' is defined in}}
  std::wcsncmp; // expected-error {{no member}} expected-note {{'std::wcsncmp' is defined in}}
  std::wcsncpy; // expected-error {{no member}} expected-note {{'std::wcsncpy' is defined in}}
  std::wcspbrk; // expected-error {{no member}} expected-note {{'std::wcspbrk' is defined in}}
  std::wcsrchr; // expected-error {{no member}} expected-note {{'std::wcsrchr' is defined in}}
  std::wcsrtombs; // expected-error {{no member}} expected-note {{'std::wcsrtombs' is defined in}}
  std::wcsspn; // expected-error {{no member}} expected-note {{'std::wcsspn' is defined in}}
  std::wcsstr; // expected-error {{no member}} expected-note {{'std::wcsstr' is defined in}}
  std::wcstod; // expected-error {{no member}} expected-note {{'std::wcstod' is defined in}}
  std::wcstof; // expected-error {{no member}} expected-note {{'std::wcstof' is defined in}} expected-note {{'std::wcstof' is a}}
  std::wcstoimax; // expected-error {{no member}} expected-note {{'std::wcstoimax' is defined in}} expected-note {{'std::wcstoimax' is a}}
  std::wcstok; // expected-error {{no member}} expected-note {{'std::wcstok' is defined in}}
  std::wcstol; // expected-error {{no member}} expected-note {{'std::wcstol' is defined in}}
  std::wcstold; // expected-error {{no member}} expected-note {{'std::wcstold' is defined in}} expected-note {{'std::wcstold' is a}}
  std::wcstoll; // expected-error {{no member}} expected-note {{'std::wcstoll' is defined in}} expected-note {{'std::wcstoll' is a}}
  std::wcstombs; // expected-error {{no member}} expected-note {{'std::wcstombs' is defined in}}
  std::wcstoul; // expected-error {{no member}} expected-note {{'std::wcstoul' is defined in}}
  std::wcstoull; // expected-error {{no member}} expected-note {{'std::wcstoull' is defined in}} expected-note {{'std::wcstoull' is a}}
  std::wcstoumax; // expected-error {{no member}} expected-note {{'std::wcstoumax' is defined in}} expected-note {{'std::wcstoumax' is a}}
  std::wcsub_match; // expected-error {{no member}} expected-note {{'std::wcsub_match' is defined in}} expected-note {{'std::wcsub_match' is a}}
  std::wcsxfrm; // expected-error {{no member}} expected-note {{'std::wcsxfrm' is defined in}}
  std::wctob; // expected-error {{no member}} expected-note {{'std::wctob' is defined in}}
  std::wctomb; // expected-error {{no member}} expected-note {{'std::wctomb' is defined in}}
  std::wctrans; // expected-error {{no member}} expected-note {{'std::wctrans' is defined in}}
  std::wctrans_t; // expected-error {{no member}} expected-note {{'std::wctrans_t' is defined in}}
  std::wctype; // expected-error {{no member}} expected-note {{'std::wctype' is defined in}}
  std::wctype_t; // expected-error {{no member}} expected-note {{'std::wctype_t' is defined in}}
  std::weak_order; // expected-error {{no member}} expected-note {{'std::weak_order' is defined in}} expected-note {{'std::weak_order' is a}}
  std::weak_ordering; // expected-error {{no member}} expected-note {{'std::weak_ordering' is defined in}} expected-note {{'std::weak_ordering' is a}}
  std::weak_ptr; // expected-error {{no member}} expected-note {{'std::weak_ptr' is defined in}} expected-note {{'std::weak_ptr' is a}}
  std::weakly_incrementable; // expected-error {{no member}} expected-note {{'std::weakly_incrementable' is defined in}} expected-note {{'std::weakly_incrementable' is a}}
  std::weibull_distribution; // expected-error {{no member}} expected-note {{'std::weibull_distribution' is defined in}} expected-note {{'std::weibull_distribution' is a}}
  std::wfilebuf; // expected-error {{no member}} expected-note {{'std::wfilebuf' is defined in}}
  std::wfilebuf; // expected-error {{no member}} expected-note {{'std::wfilebuf' is defined in}}
  std::wformat_args; // expected-error {{no member}} expected-note {{'std::wformat_args' is defined in}} expected-note {{'std::wformat_args' is a}}
  std::wformat_context; // expected-error {{no member}} expected-note {{'std::wformat_context' is defined in}} expected-note {{'std::wformat_context' is a}}
  std::wformat_parse_context; // expected-error {{no member}} expected-note {{'std::wformat_parse_context' is defined in}} expected-note {{'std::wformat_parse_context' is a}}
  std::wformat_string; // expected-error {{no member}} expected-note {{'std::wformat_string' is defined in}} expected-note {{'std::wformat_string' is a}}
  std::wfstream; // expected-error {{no member}} expected-note {{'std::wfstream' is defined in}}
  std::wfstream; // expected-error {{no member}} expected-note {{'std::wfstream' is defined in}}
  std::wifstream; // expected-error {{no member}} expected-note {{'std::wifstream' is defined in}}
  std::wifstream; // expected-error {{no member}} expected-note {{'std::wifstream' is defined in}}
  std::wios; // expected-error {{no member}} expected-note {{'std::wios' is defined in}}
  std::wios; // expected-error {{no member}} expected-note {{'std::wios' is defined in}}
  std::wios; // expected-error {{no member}} expected-note {{'std::wios' is defined in}}
  std::wiostream; // expected-error {{no member}} expected-note {{'std::wiostream' is defined in}}
  std::wiostream; // expected-error {{no member}} expected-note {{'std::wiostream' is defined in}}
  std::wiostream; // expected-error {{no member}} expected-note {{'std::wiostream' is defined in}}
  std::wispanstream; // expected-error {{no member}} expected-note {{'std::wispanstream' is defined in}} expected-note {{'std::wispanstream' is a}}
  std::wispanstream; // expected-error {{no member}} expected-note {{'std::wispanstream' is defined in}} expected-note {{'std::wispanstream' is a}}
  std::wistream; // expected-error {{no member}} expected-note {{'std::wistream' is defined in}}
  std::wistream; // expected-error {{no member}} expected-note {{'std::wistream' is defined in}}
  std::wistream; // expected-error {{no member}} expected-note {{'std::wistream' is defined in}}
  std::wistringstream; // expected-error {{no member}} expected-note {{'std::wistringstream' is defined in}}
  std::wistringstream; // expected-error {{no member}} expected-note {{'std::wistringstream' is defined in}}
  std::wmemchr; // expected-error {{no member}} expected-note {{'std::wmemchr' is defined in}}
  std::wmemcmp; // expected-error {{no member}} expected-note {{'std::wmemcmp' is defined in}}
  std::wmemcpy; // expected-error {{no member}} expected-note {{'std::wmemcpy' is defined in}}
  std::wmemmove; // expected-error {{no member}} expected-note {{'std::wmemmove' is defined in}}
  std::wmemset; // expected-error {{no member}} expected-note {{'std::wmemset' is defined in}}
  std::wofstream; // expected-error {{no member}} expected-note {{'std::wofstream' is defined in}}
  std::wofstream; // expected-error {{no member}} expected-note {{'std::wofstream' is defined in}}
  std::wospanstream; // expected-error {{no member}} expected-note {{'std::wospanstream' is defined in}} expected-note {{'std::wospanstream' is a}}
  std::wospanstream; // expected-error {{no member}} expected-note {{'std::wospanstream' is defined in}} expected-note {{'std::wospanstream' is a}}
  std::wostream; // expected-error {{no member}} expected-note {{'std::wostream' is defined in}}
  std::wostream; // expected-error {{no member}} expected-note {{'std::wostream' is defined in}}
  std::wostream; // expected-error {{no member}} expected-note {{'std::wostream' is defined in}}
  std::wostringstream; // expected-error {{no member}} expected-note {{'std::wostringstream' is defined in}}
  std::wostringstream; // expected-error {{no member}} expected-note {{'std::wostringstream' is defined in}}
  std::wosyncstream; // expected-error {{no member}} expected-note {{'std::wosyncstream' is defined in}} expected-note {{'std::wosyncstream' is a}}
  std::wosyncstream; // expected-error {{no member}} expected-note {{'std::wosyncstream' is defined in}} expected-note {{'std::wosyncstream' is a}}
  std::wprintf; // expected-error {{no member}} expected-note {{'std::wprintf' is defined in}}
  std::wregex; // expected-error {{no member}} expected-note {{'std::wregex' is defined in}} expected-note {{'std::wregex' is a}}
  std::ws; // expected-error {{no member}} expected-note {{'std::ws' is defined in}}
  std::ws; // expected-error {{no member}} expected-note {{'std::ws' is defined in}}
  std::wscanf; // expected-error {{no member}} expected-note {{'std::wscanf' is defined in}}
  std::wsmatch; // expected-error {{no member}} expected-note {{'std::wsmatch' is defined in}} expected-note {{'std::wsmatch' is a}}
  std::wspanbuf; // expected-error {{no member}} expected-note {{'std::wspanbuf' is defined in}} expected-note {{'std::wspanbuf' is a}}
  std::wspanbuf; // expected-error {{no member}} expected-note {{'std::wspanbuf' is defined in}} expected-note {{'std::wspanbuf' is a}}
  std::wspanstream; // expected-error {{no member}} expected-note {{'std::wspanstream' is defined in}} expected-note {{'std::wspanstream' is a}}
  std::wspanstream; // expected-error {{no member}} expected-note {{'std::wspanstream' is defined in}} expected-note {{'std::wspanstream' is a}}
  std::wsregex_iterator; // expected-error {{no member}} expected-note {{'std::wsregex_iterator' is defined in}} expected-note {{'std::wsregex_iterator' is a}}
  std::wsregex_token_iterator; // expected-error {{no member}} expected-note {{'std::wsregex_token_iterator' is defined in}} expected-note {{'std::wsregex_token_iterator' is a}}
  std::wssub_match; // expected-error {{no member}} expected-note {{'std::wssub_match' is defined in}} expected-note {{'std::wssub_match' is a}}
  std::wstreambuf; // expected-error {{no member}} expected-note {{'std::wstreambuf' is defined in}}
  std::wstreambuf; // expected-error {{no member}} expected-note {{'std::wstreambuf' is defined in}}
  std::wstreambuf; // expected-error {{no member}} expected-note {{'std::wstreambuf' is defined in}}
  std::wstreampos; // expected-error {{no member}} expected-note {{'std::wstreampos' is defined in}}
  std::wstreampos; // expected-error {{no member}} expected-note {{'std::wstreampos' is defined in}}
  std::wstring; // expected-error {{no member}} expected-note {{'std::wstring' is defined in}}
  std::wstring_convert; // expected-error {{no member}} expected-note {{'std::wstring_convert' is defined in}}
  std::wstring_convert; // expected-error {{no member}} expected-note {{'std::wstring_convert' is defined in}}
  std::wstring_view; // expected-error {{no member}} expected-note {{'std::wstring_view' is defined in}} expected-note {{'std::wstring_view' is a}}
  std::wstringbuf; // expected-error {{no member}} expected-note {{'std::wstringbuf' is defined in}}
  std::wstringbuf; // expected-error {{no member}} expected-note {{'std::wstringbuf' is defined in}}
  std::wstringstream; // expected-error {{no member}} expected-note {{'std::wstringstream' is defined in}}
  std::wstringstream; // expected-error {{no member}} expected-note {{'std::wstringstream' is defined in}}
  std::wsyncbuf; // expected-error {{no member}} expected-note {{'std::wsyncbuf' is defined in}} expected-note {{'std::wsyncbuf' is a}}
  std::wsyncbuf; // expected-error {{no member}} expected-note {{'std::wsyncbuf' is defined in}} expected-note {{'std::wsyncbuf' is a}}
  std::yocto; // expected-error {{no member}} expected-note {{'std::yocto' is defined in}} expected-note {{'std::yocto' is a}}
  std::yotta; // expected-error {{no member}} expected-note {{'std::yotta' is defined in}} expected-note {{'std::yotta' is a}}
  std::zepto; // expected-error {{no member}} expected-note {{'std::zepto' is defined in}} expected-note {{'std::zepto' is a}}
  std::zetta; // expected-error {{no member}} expected-note {{'std::zetta' is defined in}} expected-note {{'std::zetta' is a}}
}
