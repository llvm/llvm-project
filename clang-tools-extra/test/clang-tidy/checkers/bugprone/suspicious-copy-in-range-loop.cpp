// RUN: %check_clang_tidy %s bugprone-suspicious-copy-in-range-loop %t

std::vector<std::string> some_strings;
for (auto x : some_strings) {
	std::cout << x << "\n";
}
// CHECK-MESSAGES: foo
