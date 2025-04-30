#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test bi06  ########


bi06: run
	

build:  $(SRC)/bi06.f
	-$(RM) bi06.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(SRC)/bi06.f -o bi06.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bi06.$(OBJX) check.$(OBJX)  $(LIBS) -o bi06.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test bi06
	bi06.$(EXESUFFIX)

verify: ;

bi06.run: run

