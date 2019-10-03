namespace std {
class basic_string {};
typedef basic_string string;
} // namespace std

std::string foo(); //  // CHECK: std::new_string foo();

// RUN: clang-rename -offset=93 -new-name=new_string %s --  | sed 's,//.*,,' | FileCheck %s
