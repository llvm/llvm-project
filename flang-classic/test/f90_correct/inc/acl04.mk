#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test acl04  ########


acl04: run


build:  $(SRC)/acl04.f90
	-$(RM) acl04.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC)  -c $(FFLAGS) $(LDFLAGS) $(SRC)/acl04.f90 -o acl04.$(OBJX)
	-$(FC)  $(FFLAGS) $(LDFLAGS) acl04.$(OBJX) check.$(OBJX) $(LIBS) -o acl04.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test acl04
	acl04.$(EXESUFFIX)

verify: ;

