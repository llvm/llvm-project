#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
do19: do19.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN4) ./a.$(EXESUFFIX) $(LOG)
do19.$(OBJX): $(SRC)/do19.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/do19.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) do19.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: do19
run: ;
