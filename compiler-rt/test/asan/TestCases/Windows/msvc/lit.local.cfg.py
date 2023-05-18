import re

if not re.match(r".*-windows-msvc$", config.target_triple):
    config.unsupported = True
