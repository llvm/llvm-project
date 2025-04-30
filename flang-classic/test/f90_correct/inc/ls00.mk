#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls00  ########


ls00: run
	

build:  $(SRC)/ls00.f
	-$(RM) ls00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls00.f -o ls00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls00.$(OBJX) check.$(OBJX) $(LIBS) -o ls00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls00
	ls00.$(EXESUFFIX)

verify: ;

ls00.run: run

