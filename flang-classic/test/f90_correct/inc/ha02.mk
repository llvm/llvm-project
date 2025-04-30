#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test ha02  ########
SHELL := /bin/bash

ha02: run


build:  $(SRC)/ha02.f90
	-$(RM) ha02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ha02.f90 -o ha02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ha02.$(OBJX) $(LIBS) -o ha02.$(EXESUFFIX)

#
# NOTE: Cygwin will detect the high-bit (sign bit) set and return a value 
# of 127.  We capture this case for Cygwin terminals in the conditional below.
#
run:
	@echo ------------------------------------ executing test ha02
	@ha02.$(EXESUFFIX) 1>exitstr 2>&1; \
	stat=`echo $$?`; \
	cat exitstr ; \
	echo ------------------------------------- ; \
	exitstr=`cat exitstr | tail -n 1 | tr -d " \n\r"`; \
	os=`uname -o`; \
	if [[ "$$os" = "Cygwin" && "$$stat" = "127" && "$$exitstr" = "-42" ]] ; \
		 then echo "$$stat $$exitstr -- 1 tests completed. 1 tests PASSED. 0 tests failed."; \
	elif [[ "$$stat" = "214" && "$$exitstr" = "-42" ]] ; \
		 then echo "$$stat $$exitstr -- 1 tests completed. 1 tests PASSED. 0 tests failed."; \
	else \
		echo "$$stat $$exitstr -- 1 tests completed. 0 tests PASSED. 1 tests failed."; \
	fi;
	@$(RM) exitstr;

verify: ;

ha02.run: run
