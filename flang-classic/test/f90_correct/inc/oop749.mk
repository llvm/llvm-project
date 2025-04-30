#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test oop749  ########


oop749: run
	

build:  $(SRC)/oop749.f90
	-$(RM) oop749.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop749.f90 -o oop749.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop749.$(OBJX) check.$(OBJX) $(LIBS) -o oop749.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test oop749
	oop749.$(EXESUFFIX)

verify: ;

