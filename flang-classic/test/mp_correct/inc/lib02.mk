#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
lib02: lib02.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
lib02.$(OBJX): $(SRC)/lib02.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/lib02.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) lib02.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: lib02
run: ;
