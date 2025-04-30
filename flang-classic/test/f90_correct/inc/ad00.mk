#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ad00  ########


ad00: run
	

build:  $(SRC)/ad00.f
	-$(RM) ad00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ad00.f -o ad00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ad00.$(OBJX) check.$(OBJX) $(LIBS) -o ad00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ad00
	ad00.$(EXESUFFIX)

verify: ;

ad00.run: run

