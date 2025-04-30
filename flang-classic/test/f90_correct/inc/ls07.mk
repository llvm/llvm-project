#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls07  ########


ls07: run
	

build:  $(SRC)/ls07.f
	-$(RM) ls07.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls07.f -o ls07.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls07.$(OBJX) check.$(OBJX) $(LIBS) -o ls07.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls07
	ls07.$(EXESUFFIX)

verify: ;

ls07.run: run

