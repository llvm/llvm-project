
# RUN: %{python} %{libcxx-dir}/../clang-tools-extra/clang-tidy/tool/run-clang-tidy.py -clang-tidy-binary %{clang-tidy} -warnings-as-errors \* -source-filter=".*libcxx/src.*" -quiet -p %{bin-dir}/..
