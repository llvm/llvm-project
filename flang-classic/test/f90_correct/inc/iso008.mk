#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#
########## Make rule for test iso008  ########


iso008: run
	

build:  $(SRC)/iso008.f90
	-$(RM) iso008.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/c_iso008.c -o c_iso008.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/iso008.f90 -o iso008.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) iso008.$(OBJX) c_iso008.$(OBJX) $(LIBS) -o iso008.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test iso008
	iso008.$(EXESUFFIX)

verify: ;

