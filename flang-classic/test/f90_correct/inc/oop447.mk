#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop447  ########


oop447: run
	

build:  $(SRC)/oop447.f90
	-$(RM) oop447.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop447.f90 -o oop447.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop447.$(OBJX) check.$(OBJX) $(LIBS) -o oop447.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop447
	oop447.$(EXESUFFIX)

verify: ;

