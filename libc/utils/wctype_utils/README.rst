This folder contains utility scripts for wide character in Python 3 for
generating the necessary data used by internal implementation of ``wctype``
utils. These scripts are meant to be run manually by the maintainers when the
data needs to be updated or a new version of unicode data are released. The
generated data and files are then checked into the repository by the maintainers
and built with the internal helper utils found in ``libc/src/__support/wctype``.
Manual modification of the generated files is prohibited.
