#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qn08  ########


rio5: run
	

build:  $(SRC)/rio5.f08
	-$(RM) rio5.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/rio5.f08 -o rio5.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) rio5.$(OBJX) check.$(OBJX) $(LIBS) -o rio5.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qn08
	rio5.$(EXESUFFIX)

verify: ;

rio5.run: run

