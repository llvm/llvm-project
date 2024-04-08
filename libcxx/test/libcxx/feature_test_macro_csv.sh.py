# RUN: %{python} %s %{libcxx-dir}/utils %{libcxx-dir}/utils/data/feature_test_macros/test_data.json

import sys
import json

sys.path.append(sys.argv[1])
from generate_feature_test_macro_components import get_table


data = json.load(open(f"{sys.argv[2]}"))
table = get_table(data)

expected = {
    "__cpp_lib_any": {
        "c++17": "201606L",
        "c++20": "201606L",
        "c++23": "201606L",
        "c++26": "201606L",
    },
    "__cpp_lib_barrier": {"c++20": "201907L", "c++23": "201907L", "c++26": "201907L"},
    "__cpp_lib_format": {
        "c++20": "",
        "c++23": "",
        "c++26": "",
    },
    "__cpp_lib_variant": {
        "c++17": "202102L",
        "c++20": "202102L",
        "c++23": "202102L",
        "c++26": "202102L",
    },
}


assert table == expected, f"expected\n{expected}\n\nresult\n{table}"
