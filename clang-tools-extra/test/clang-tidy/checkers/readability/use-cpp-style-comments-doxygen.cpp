// RUN: %check_clang_tidy -std=c++11 %s readability-use-cpp-style-comments %t -config="{CheckOptions: [{key: readability-use-cpp-style-comments.ExcludeDoxygenStyleComments, value: true}]}"

// Tests for Doxygen comments with ExcludeDoxygenStyleComments enabled
/**
 * This is a Doxygen comment for a function.
 * It should NOT be transformed.
 */
void doxygenFunction1();

/*!
 * This is another Doxygen-style comment.
 * It should also NOT be transformed.
 */
void doxygenFunction2();

/**
 * Multiline Doxygen comment describing parameters.
 *
 * @param x The first parameter.
 * @param y The second parameter.
 * @return A result value.
 */
int doxygenFunctionWithParams(int x, int y);

/*******************************
 * Non-Doxygen block comments without markers
 *******************************/
void DoxygenBlock();

/*!
 * This is a single-line Doxygen comment.
 * Should NOT be transformed.
 */
void singleLineDoxygen();
