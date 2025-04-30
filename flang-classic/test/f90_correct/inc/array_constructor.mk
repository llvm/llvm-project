#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

build:  $(SRC)/array_constructor.f90
	-$(RM) array_constructor.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	-$(RM) $(OBJ)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	@echo ------------------------------------ building test $@

	# Do two compilations: one with -default-integer-8, which caused an ICE
	# before https://github.com/flang-compiler/flang/issues/745 was fixed.
	$(FC)                     $(FFLAGS) $(LDFLAGS) $(SRC)/array_constructor.f90 check.$(OBJX) -o array_constructor.$(EXESUFFIX)
	$(FC) -fdefault-integer-8 $(FFLAGS) $(LDFLAGS) $(SRC)/array_constructor.f90 check.$(OBJX) -o array_constructor.defaultint8.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test array_constructor
	array_constructor.$(EXESUFFIX)
	array_constructor.defaultint8.$(EXESUFFIX)

verify: ;

array_constructor.run: run
