#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop431  ########


oop431: run
	

build:  $(SRC)/oop431.f90
	-$(RM) oop431.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop431.f90 -o oop431.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop431.$(OBJX) check.$(OBJX) $(LIBS) -o oop431.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop431
	oop431.$(EXESUFFIX)

verify: ;

