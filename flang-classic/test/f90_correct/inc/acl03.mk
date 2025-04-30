#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test acl03  ########


acl03: run
	

build:  $(SRC)/acl03.f90
	-$(RM) acl03.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC)  -c $(FFLAGS) $(LDFLAGS) $(SRC)/acl03.f90 -o acl03.$(OBJX)
	-$(FC)  $(FFLAGS) $(LDFLAGS) acl03.$(OBJX) check.$(OBJX) $(LIBS) -o acl03.$(EXESUFFIX)


run: 
	@echo ------------------------------------ executing test acl03
	acl03.$(EXESUFFIX)

verify: ;

