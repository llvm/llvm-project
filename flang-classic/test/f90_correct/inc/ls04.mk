#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls04  ########


ls04: run
	

build:  $(SRC)/ls04.f
	-$(RM) ls04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls04.f -o ls04.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls04.$(OBJX) check.$(OBJX) $(LIBS) -o ls04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls04
	ls04.$(EXESUFFIX)

verify: ;

ls04.run: run

