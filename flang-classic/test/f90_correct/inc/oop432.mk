#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop432  ########


oop432: run
	

build:  $(SRC)/oop432.f90
	-$(RM) oop432.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop432.f90 -o oop432.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop432.$(OBJX) check.$(OBJX) $(LIBS) -o oop432.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop432
	oop432.$(EXESUFFIX)

verify: ;

