import os
import sys

pkgRelDir = sys.argv[1]
pkgFiles = sys.argv[2:]

getFileName = lambda f: os.path.splitext(os.path.basename(f))[0]
importNames = ', '.join('"{}"'.format(getFileName(f)) for f in pkgFiles)

script = """__all__ = [{import_names}]
for x in __all__:
  __import__('lldb.{pkg_name}.' + x)
""".format(import_names=importNames, pkg_name=pkgRelDir.replace("/", "."))

pkgIniFile = os.path.normpath(os.path.join(pkgRelDir, "__init__.py"))
with open(pkgIniFile, "w") as f:
    f.write(script)
